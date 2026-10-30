"""Phase 2: look inside. Does learning to generate change what the hidden layers encode?

Compares the networks in results/final/ (bPC at each alpha_gen, discPC = alpha_gen 0,
and the backprop MLP) on the fixed 1,000-image probe set:

  A  reproduction table: test accuracy and generation RMSE vs the paper
     (RMSE reported in the paper's units, i.e. on a [0, 1] pixel scale = ours / 2;
     see docs/REPLICATION.md for why)
  B  1 - CKA per layer, bPC vs discPC, against the seed floor; final and at matched accuracy
  C  the abstraction ladder: reconstruct the image from each layer's inferred state;
     how close to the image itself, how close to its class average, and whether the
     reconstruction still identifies *this* image among the 100 probe images of its class
  D  linear read-out of the label from each layer
  E  projective fields of single units (pictures only)
  F  profile: CKA of each layer with the pixels and with the one-hot label

    PYTHONPATH=. ../PCN-Version2/.venv/bin/python scripts/inside.py
Writes results/final/inside.json and results/figures/*.png.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcn.bpc import BPC, BPCConfig
from pcn.bpc_data import mnist_split
from pcn.metrics import linear_cka
from pcn.stats import compare_to_floor

ROOT = Path(__file__).resolve().parent.parent
import os
FINAL = Path(os.environ.get("INSIDE_DIR", ROOT / "results" / "final"))
FIG = Path(os.environ.get("INSIDE_FIG", ROOT / "results" / "figures"))
PAPER = {"bPC": (98.10, 0.05, 0.0581, 0.0004), "discPC": (98.43, 0.01, 0.3133, 0.0224)}
LAYERS = ("L1", "L2", "L3")
GREY, BLUE, ORANGE, GREEN = "#8a8a8a", "#2f6fb0", "#d9822b", "#3a9a5b"


SHORT = {"discPC": "discPC\n(α_gen 0)", "bPC": "bPC\n(α_gen 10⁻⁴)", "bPC α_gen=0.001": "bPC\n(α_gen 10⁻³)", "bPC α_gen=0.01": "bPC\n(α_gen 10⁻²)"}


def cond_name(a_gen: float) -> str:
    return "discPC" if a_gen == 0 else ("bPC" if a_gen == 1e-4 else f"bPC α_gen={a_gen:g}")


def load_runs() -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for p in sorted(FINAL.glob("agen*.json")):
        r = json.loads(p.read_text())
        r["probe"] = dict(np.load(FINAL / f"{r['name']}__probe.npz"))
        runs.setdefault(cond_name(r["config"]["a_gen"]), []).append(r)
    return runs


def load_net(r: dict) -> BPC:
    cfg = dict(r["config"])
    cfg["sizes"] = tuple(cfg["sizes"])
    net = BPC(BPCConfig(**cfg))
    net.load_state_dict(torch.load(FINAL / f"{r['name']}.pt")["state"])
    return net.eval()


def states(r: dict, ep: int | None = None) -> dict[str, torch.Tensor]:
    ep = ep or r["config"]["epochs"]
    return {l: torch.from_numpy(r["probe"][f"e{ep}_{l}"].astype(np.float32)) for l in LAYERS}


def snap_acc(r: dict) -> dict[int, float]:
    return {h["epoch"]: h["test_acc"] for h in r["history"] if "test_acc" in h}


def mean_sem(v) -> tuple[float, float]:
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


# ---------------------------------------------------------------------------- B
def cka_div(A: dict, B: dict, l: str) -> float:
    return 1.0 - linear_cka(A[l], B[l])


def cka_block(ref: list[dict], other: list[dict], other_eps: list[int] | None = None) -> dict:
    """1 - CKA: floor = pairs within `ref` (and within `other`), test = ref x other pairs."""
    R = [states(r) for r in ref]
    O = [states(r, e) for r, e in zip(other, other_eps or [None] * len(other))]
    out = {}
    for l in LAYERS:
        floor = [cka_div(a, b, l) for a, b in itertools.combinations(R, 2)]
        floor += [cka_div(a, b, l) for a, b in itertools.combinations(O, 2)]
        cross = [cka_div(a, b, l) for a in R for b in O]
        e = compare_to_floor(np.array(cross), np.array(floor), n_perm=5000).to_dict()
        out[l] = {"floor": float(np.mean(floor)), "cross": float(np.mean(cross)),
                  "ratio": float(np.mean(cross) / np.mean(floor)), "d": e["cohens_d"],
                  "auc": e["auc"], "p": e["perm_p"],
                  "every_cross_above_every_floor": bool(min(cross) > max(floor))}
    return out


def matched_epochs(other: list[dict], target_acc: float) -> list[int]:
    """For each network, the snapshot whose test accuracy is closest to target_acc."""
    return [min(snap_acc(r).items(), key=lambda kv: abs(kv[1] - target_acc))[0] for r in other]


# ---------------------------------------------------------------------------- C
@torch.no_grad()
def ladder(net: BPC, xp: torch.Tensor, yp: torch.Tensor, means: torch.Tensor) -> dict:
    x_inf = net.infer(xp)
    out, imgs = {}, {}
    same = (yp[:, None] == yp[None, :])
    for k in range(1, net.L + 1):
        rec = net.reconstruct_from(x_inf, k)
        imgs[k] = rec
        d_img = (rec - xp).pow(2).mean(1).sqrt()
        d_cls = (rec - means[yp]).pow(2).mean(1).sqrt()
        D = torch.cdist(rec, xp)                        # recon i vs probe image j
        D_same = D.masked_fill(~same, float("inf"))
        inst = (D_same.argmin(1) == torch.arange(len(yp))).float().mean()
        cls = (torch.cdist(rec, means).argmin(1) == yp).float().mean()
        out[f"L{k}"] = {"rmse_to_image": float(d_img.mean()), "rmse_to_class_mean": float(d_cls.mean()),
                        "instance_id": float(inst), "class_id": float(cls)}
    return out, imgs


# ---------------------------------------------------------------------------- D
def decode(S: dict, y: np.ndarray) -> dict:
    out = {}
    for l in LAYERS:
        X = S[l].numpy() if torch.is_tensor(S[l]) else S[l]
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        out[l] = float(cross_val_score(clf, X, y, cv=5).mean())
    return out


# ---------------------------------------------------------------------------- E
@torch.no_grad()
def projective_fields(net: BPC, x_inf: list[torch.Tensor], k: int, n: int = 8, scale: float = 3.0):
    base = x_inf[k].mean(0, keepdim=True)
    sd = x_inf[k].std(0)
    units = x_inf[k].var(0).argsort(descending=True)[:n]
    pfs = []
    for u in units:
        hi = base.clone()
        hi[0, u] += scale * sd[u]
        a = net.sweep_dn([None] * k + [hi], k)[0]
        b = net.sweep_dn([None] * k + [base], k)[0]
        pfs.append((a - b)[0].numpy())
    return np.stack(pfs), units.numpy()


# ---------------------------------------------------------------------------- main
def main() -> None:
    torch.set_num_threads(6)
    FIG.mkdir(parents=True, exist_ok=True)
    d = mnist_split()
    xp, yp = d.probe
    means = d.class_means("test")
    runs = load_runs()
    order = [c for c in ("discPC", "bPC", "bPC α_gen=0.001", "bPC α_gen=0.01") if c in runs]
    res: dict = {"conditions": order, "n": {c: len(runs[c]) for c in order}}

    # A ---------------------------------------------------------------------
    res["A"] = {}
    for c in order:
        acc = mean_sem([100 * r["test_acc"] for r in runs[c]])
        rm = mean_sem([r["test_rmse"] / 2 for r in runs[c]])     # paper units ([0, 1] pixels)
        res["A"][c] = {"acc": acc, "rmse": rm, "paper": PAPER.get(c)}
    bp = json.loads((FINAL / "bp.json").read_text())
    res["A"]["BP"] = {"acc": mean_sem([100 * s["test_acc"] for s in bp["seeds"]]), "paper": (98.48, 0.10)}
    print(json.dumps(res["A"], indent=1), flush=True)

    # B ---------------------------------------------------------------------
    res["B"] = {}
    disc = runs["discPC"]
    for c in order[1:]:
        tgt = float(np.mean([r["test_acc"] for r in runs[c]]))
        eps = matched_epochs(disc, tgt)
        res["B"][c] = {
            "final": cka_block(runs[c], disc),
            "matched": cka_block(runs[c], disc, eps),
            "matched_epochs": eps,
            "matched_disc_acc": float(np.mean([snap_acc(r)[e] for r, e in zip(disc, eps)])),
            "target_acc": tgt,
        }
    # similarity to backprop, per layer
    bp_states = [{l: torch.from_numpy(np.load(FINAL / f"bp_s{s}__probe.npz")[l].astype(np.float32)) for l in LAYERS}
                 for s in range(len(bp["seeds"]))]
    res["B"]["vs_BP"] = {c: {l: float(np.mean([cka_div(states(r), b, l) for r in runs[c] for b in bp_states]))
                             for l in LAYERS} for c in order}
    res["B"]["BP_floor"] = {l: float(np.mean([cka_div(a, b, l) for a, b in itertools.combinations(bp_states, 2)]))
                            for l in LAYERS}
    print(json.dumps(res["B"], indent=1), flush=True)

    # C, D, E -------------------------------------------------------------
    res["C"], res["D"] = {}, {}
    ladder_imgs = {}
    for c in order:
        rows, ys = [], yp.numpy()
        for i, r in enumerate(runs[c]):
            net = load_net(r)
            lad, imgs = ladder(net, xp, yp, means)
            rows.append(lad)
            if i == 0:
                ladder_imgs[c] = imgs
                if c == "bPC":
                    x_inf = net.infer(xp)
                    pf = {k: projective_fields(net, x_inf, k) for k in (1, 2)}
        res["C"][c] = {l: {m: mean_sem([row[l][m] for row in rows]) for m in rows[0][l]} for l in rows[0]}
        dec = [decode(states(r), ys) for r in runs[c]]
        res["D"][c] = {l: mean_sem([x[l] for x in dec]) for l in LAYERS}
    dec = [decode(b, yp.numpy()) for b in bp_states]
    res["D"]["BP"] = {l: mean_sem([x[l] for x in dec]) for l in LAYERS}
    print(json.dumps({"C": res["C"], "D": res["D"]}, indent=1), flush=True)

    # F ---------------------------------------------------------------------
    onehot = torch.nn.functional.one_hot(yp, 10).float()
    prof = lambda S: {l: (linear_cka(S[l], xp), linear_cka(S[l], onehot)) for l in LAYERS}
    res["F"] = {"pixels_vs_label": linear_cka(xp, onehot)}
    for c in order:
        rows = [prof(states(r)) for r in runs[c]]
        res["F"][c] = {l: {"pixel": mean_sem([x[l][0] for x in rows]), "label": mean_sem([x[l][1] for x in rows])} for l in LAYERS}
    rows = [prof(b) for b in bp_states]
    res["F"]["BP"] = {l: {"pixel": mean_sem([x[l][0] for x in rows]), "label": mean_sem([x[l][1] for x in rows])} for l in LAYERS}
    print(json.dumps(res["F"], indent=1), flush=True)
    (FINAL / "inside.json").write_text(json.dumps(res, indent=1))

    figures(runs, order, res, ladder_imgs, pf, xp, yp, means)


# ---------------------------------------------------------------------------- figures
def _img(ax, v, title=None, cmap="gray", sym=False):
    v = np.asarray(v).reshape(28, 28)
    if sym:
        m = np.abs(v).max() + 1e-9
        ax.imshow(v, cmap="RdBu_r", vmin=-m, vmax=m)
    else:
        ax.imshow(v, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def figures(runs, order, res, ladder_imgs, pf, xp, yp, means):
    # 1. generated digits per condition
    fig, ax = plt.subplots(len(order) + 1, 10, figsize=(10, 1.1 * (len(order) + 1)))
    for j in range(10):
        _img(ax[0, j], means[j], str(j))
    ax[0, 0].set_ylabel("class mean", fontsize=8)
    for i, c in enumerate(order, 1):
        g = runs[c][0]["probe"]["generated"]
        for j in range(10):
            _img(ax[i, j], g[j])
        ax[i, 0].set_ylabel(SHORT[c], fontsize=8)
    fig.suptitle("Label → image: what each network generates (seed 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "generated.png", dpi=150)
    plt.close(fig)

    # 2. abstraction ladder: 4 probe digits, reconstruct from each layer
    pick = [int(np.flatnonzero(yp.numpy() == c)[0]) for c in (3, 8, 5, 2)]
    for c in ("bPC", "discPC"):
        imgs = ladder_imgs[c]
        fig, ax = plt.subplots(len(pick), 5, figsize=(6, 1.3 * len(pick)))
        for i, p in enumerate(pick):
            _img(ax[i, 0], xp[p], "input" if i == 0 else None)
            for k in (1, 2, 3):
                _img(ax[i, k], imgs[k][p], f"from L{k}" if i == 0 else None)
            _img(ax[i, 4], means[yp[p]], "class mean" if i == 0 else None)
        fig.suptitle(f"{c}: what each layer keeps of the image", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"ladder_{c}.png", dpi=150)
        plt.close(fig)

    # 3. ladder metrics (bPC vs discPC)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for c, col in (("discPC", GREY), ("bPC", BLUE)):
        C = res["C"][c]
        ls = list(C)
        x = np.arange(len(ls))
        ax[0].errorbar(x, [C[l]["rmse_to_image"][0] for l in ls], [C[l]["rmse_to_image"][1] for l in ls], marker="o", color=col, label=f"{c}: to the image")
        ax[0].errorbar(x, [C[l]["rmse_to_class_mean"][0] for l in ls], [C[l]["rmse_to_class_mean"][1] for l in ls], marker="s", ls="--", color=col, label=f"{c}: to its class mean")
        ax[1].errorbar(x, [100 * C[l]["instance_id"][0] for l in ls], [100 * C[l]["instance_id"][1] for l in ls], marker="o", color=col, label=c)
        ax[2].errorbar(x, [100 * C[l]["class_id"][0] for l in ls], [100 * C[l]["class_id"][1] for l in ls], marker="o", color=col, label=c)
    for a, t in zip(ax, ("reconstruction RMSE", "picks out *this* image (%)\n(chance 1%)", "nearest class mean is right (%)")):
        a.set_xticks(range(3), ["from L1", "from L2", "from L3 (label)"])
        a.set_title(t, fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    ax[1].axhline(1, color="k", lw=0.5, ls=":")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "ladder_metrics.png", dpi=150)
    plt.close(fig)

    # 4. CKA: bPC vs discPC against the seed floor
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    B = res["B"]["bPC"]
    for a, key, t in ((ax[0], "final", "both fully trained"), (ax[1], "matched", "discPC at matched accuracy")):
        x = np.arange(len(LAYERS))
        a.bar(x - 0.18, [B[key][l]["floor"] for l in LAYERS], 0.36, color=GREY, label="seed floor (same recipe)")
        a.bar(x + 0.18, [B[key][l]["cross"] for l in LAYERS], 0.36, color=BLUE, label="bPC vs discPC")
        for i, l in enumerate(LAYERS):
            a.text(i + 0.18, B[key][l]["cross"], f"d={B[key][l]['d']:.1f}", ha="center", va="bottom", fontsize=7)
        a.set_xticks(x, ["L1", "L2", "L3 (label)"])
        a.set_title(t, fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("1 − CKA (how different)")
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "cka.png", dpi=150)
    plt.close(fig)

    # 5. projective fields
    fig, ax = plt.subplots(2, 8, figsize=(9, 3.0))
    for i, k in enumerate((1, 2)):
        P, units = pf[k]
        for j in range(8):
            _img(ax[i, j], P[j], f"L{k} u{units[j]}", sym=True)
    fig.suptitle("bPC: what single units draw (red = brighter, blue = darker)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "projective_fields.png", dpi=150)
    plt.close(fig)
    print("figures ->", FIG, flush=True)


if __name__ == "__main__":
    main()
