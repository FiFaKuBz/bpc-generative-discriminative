"""Bidirectional predictive coding (bPC), following Oliviers, Tang & Bogacz (ICLR 2026).

Written from the paper's main text (§3, eqs 3-6) and Appendix A, so that the
MNIST experiment of §4.1 can be reproduced. ``model.py`` (PCN-Version2) is a
different formulation and is left untouched.

Layers  x[0] .. x[L]:  x[0] = image (always clamped), x[L] = label.

Energy (eq. 3), summed over units, one value per sample:

    E = sum_{l=0}^{L-1} a_gen/2  ||x[l] - g_l(x[l+1])||^2      top-down (generative)
      + sum_{l=1}^{L}   a_disc/2 ||x[l] - d_l(x[l-1])||^2      bottom-up (discriminative)

Prediction form (``form``):
    "fW"  f(W x + b), as in Appendix A.2: identity on the bottom-up prediction of
          the label, tanh on the top-down prediction of the pixels, ``act``
          everywhere else.
    "Wf"  W f(x) + b, as written in eq. 3 and in the authors' tutorial
          (5_bidirectional_pc.ipynb): ``act`` is applied to latent layers only,
          never to the clamped image or label.

Inference: SGD with momentum on the free states (A.3), after a feedforward
sweep in the direction of prediction. Learning: one AdamW step on the same
energy after inference (eq. 6, Algorithm 1). Every gradient is local: the
autograd calls below only differentiate E with respect to quantities that are
leaves of a single layer pair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTS = {
    "leaky_relu": F.leaky_relu,
    "tanh": torch.tanh,
    "gelu": F.gelu,
    "relu": F.relu,
    "identity": lambda z: z,
}


@dataclass
class BPCConfig:
    sizes: tuple[int, ...] = (784, 256, 256, 10)   # A.2: [784, 256, 256, 10]
    act: str = "leaky_relu"
    form: str = "fW"
    a_gen: float = 1e-4                            # 0 -> discPC (top-down weights unused)
    a_disc: float = 1.0
    T: int = 8                                     # inference steps per training batch
    T_eval: int = 100                              # inference steps at evaluation
    lr_x: float = 0.01
    mom_x: float = 0.0
    lr_p: float = 1e-4
    wd: float = 5e-3
    epochs: int = 25
    batch: int = 256
    seed: int = 0
    device: str = "cpu"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sizes"] = list(self.sizes)
        return d


class BPC(nn.Module):
    def __init__(self, cfg: BPCConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        s = cfg.sizes
        self.L = len(s) - 1
        self.f = ACTS[cfg.act]
        # up[l-1] predicts x[l] from x[l-1]  (l = 1..L)
        self.up = nn.ModuleList([nn.Linear(s[l - 1], s[l]) for l in range(1, self.L + 1)])
        # dn[l] predicts x[l] from x[l+1]    (l = 0..L-1)
        self.dn = nn.ModuleList([nn.Linear(s[l + 1], s[l]) for l in range(self.L)])
        self.to(cfg.device)
        params = list(self.up.parameters())
        if cfg.a_gen > 0:
            params += list(self.dn.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg.lr_p, weight_decay=cfg.wd)

    # ------------------------------------------------------------------ predictions
    def pred_up(self, l: int, src: torch.Tensor) -> torch.Tensor:
        if self.cfg.form == "fW":
            z = self.up[l - 1](src)
            return z if l == self.L else self.f(z)
        return self.up[l - 1](src if l - 1 == 0 else self.f(src))

    def pred_dn(self, l: int, src: torch.Tensor) -> torch.Tensor:
        if self.cfg.form == "fW":
            z = self.dn[l](src)
            return torch.tanh(z) if l == 0 else self.f(z)
        return self.dn[l](src if l + 1 == self.L else self.f(src))

    # ------------------------------------------------------------------ energy
    def energy(self, x: list[torch.Tensor], top: int | None = None) -> torch.Tensor:
        """Per-sample energy (B,). ``top`` restricts it to layers 0..top."""
        top = self.L if top is None else top
        E = x[0].new_zeros(x[0].shape[0])
        if self.cfg.a_disc > 0:
            for l in range(1, top + 1):
                E = E + 0.5 * self.cfg.a_disc * (x[l] - self.pred_up(l, x[l - 1])).pow(2).sum(1)
        if self.cfg.a_gen > 0:
            for l in range(0, top):
                E = E + 0.5 * self.cfg.a_gen * (x[l] - self.pred_dn(l, x[l + 1])).pow(2).sum(1)
        return E

    def relax(self, x, free, T, top=None):
        """SGD with momentum on the free layers (each sample independently)."""
        x = [xi.detach().clone() for xi in x]
        vel = {l: torch.zeros_like(x[l]) for l in free}
        lr, mom = self.cfg.lr_x, self.cfg.mom_x
        with torch.enable_grad():
            for _ in range(T):
                for l in free:
                    x[l].requires_grad_(True)
                E = self.energy(x, top).sum()
                grads = torch.autograd.grad(E, [x[l] for l in free])
                for l, g in zip(free, grads):
                    vel[l] = mom * vel[l] + g
                    x[l] = (x[l] - lr * vel[l]).detach()
        return x

    # ------------------------------------------------------------------ sweeps
    def sweep_up(self, img: torch.Tensor) -> list[torch.Tensor]:
        x = [img]
        for l in range(1, self.L + 1):
            x.append(self.pred_up(l, x[l - 1]))
        return x

    def sweep_dn(self, x: list[torch.Tensor], start: int) -> list[torch.Tensor]:
        """Overwrite x[start-1] .. x[0] by top-down predictions from x[start]."""
        x = list(x)
        for l in range(start - 1, -1, -1):
            x[l] = self.pred_dn(l, x[l + 1])
        return x

    # ------------------------------------------------------------------ learning
    def train_batch(self, img: torch.Tensor, y1h: torch.Tensor) -> float:
        with torch.no_grad():
            x = self.sweep_up(img)
            x[self.L] = y1h
        x = self.relax(x, list(range(1, self.L)), self.cfg.T)
        loss = self.energy(x).mean()                    # (1/B) sum_b dE_b/dtheta
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return loss.item()

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def infer(self, img: torch.Tensor, T: int | None = None) -> list[torch.Tensor]:
        """Discriminative inference: image clamped, everything else free (A.5)."""
        x = self.sweep_up(img)
        return self.relax(x, list(range(1, self.L + 1)), self.cfg.T_eval if T is None else T)

    @torch.no_grad()
    def classify(self, img: torch.Tensor) -> torch.Tensor:
        return self.infer(img)[self.L].argmax(1)

    @torch.no_grad()
    def generate(self, y1h: torch.Tensor, T: int | None = None) -> torch.Tensor:
        """Generative inference: label clamped, everything below free (A.5).

        bPC: top-down sweep, then inference. discPC (a_gen = 0) has no top-down
        pathway, so as in the paper x[0] starts at zero and the latents are set
        by a bottom-up sweep.
        """
        T = self.cfg.T_eval if T is None else T
        B = y1h.shape[0]
        if self.cfg.a_gen > 0:
            x = [None] * (self.L + 1)
            x[self.L] = y1h
            x = self.sweep_dn(x, self.L)
        else:
            x = self.sweep_up(y1h.new_zeros(B, self.cfg.sizes[0]))
            x[self.L] = y1h
        return self.relax(x, list(range(0, self.L)), T)[0]

    @torch.no_grad()
    def reconstruct_from(self, x_inf: list[torch.Tensor], k: int, T: int | None = None) -> torch.Tensor:
        """What layer k 'knows': clamp x[k] at its inferred state, sweep down, and
        relax layers 0..k-1 on the sub-network below k. Returns the image layer."""
        T = self.cfg.T_eval if T is None else T
        if self.cfg.a_gen > 0:
            x = [None] * (k + 1)
            x[k] = x_inf[k]
            x = self.sweep_dn(x, k)
        else:                                   # discPC: same start as generate()
            x = self.sweep_up(x_inf[k].new_zeros(x_inf[k].shape[0], self.cfg.sizes[0]))[: k + 1]
            x[k] = x_inf[k]
        return self.relax(x, list(range(0, k)), T, top=k)[0]
