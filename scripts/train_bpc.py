"""Train one bPC (or discPC with --a_gen 0) network on MNIST, as in bPC §4.1.

    PYTHONPATH=. ../PCN-Version2/.venv/bin/python scripts/train_bpc.py --tag pilot --seed 0

Writes results/<tag>/<name>.json (config, per-epoch val accuracy / RMSE, final
test scores) and, with --save, the final weights and the probe-set states of
every layer at the snapshot epochs (for the matched-accuracy inside analysis).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcn.bpc import BPC, BPCConfig
from pcn.bpc_data import mnist_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
SNAPSHOTS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 25)


@torch.no_grad()
def accuracy(net: BPC, x, y, bs: int = 1000) -> float:
    return float(sum((net.classify(x[i:i + bs]) == y[i:i + bs]).sum() for i in range(0, len(y), bs)) / len(y))


@torch.no_grad()
def gen_rmse(net: BPC, means: torch.Tensor) -> tuple[float, torch.Tensor]:
    """A.5: RMSE between the image generated for each label and that class's mean."""
    img = net.generate(torch.eye(10, device=means.device))
    return float((img - means).pow(2).mean().sqrt()), img


@torch.no_grad()
def probe_states(net: BPC, xp) -> dict:
    x = net.infer(xp)
    return {f"L{l}": x[l].cpu().numpy().astype(np.float16) for l in range(1, net.L + 1)}


def name_of(cfg: BPCConfig) -> str:
    return (f"agen{cfg.a_gen:g}_{cfg.act}_{cfg.form}_lrx{cfg.lr_x:g}_m{cfg.mom_x:g}"
            f"_lrp{cfg.lr_p:g}_wd{cfg.wd:g}_T{cfg.T}_e{cfg.epochs}_s{cfg.seed}")


def run(cfg: BPCConfig, tag: str, save: bool, eval_every: int = 1, verbose: bool = True) -> dict:
    out = RESULTS / tag
    out.mkdir(parents=True, exist_ok=True)
    name = name_of(cfg)
    torch.set_num_threads(2)
    d = mnist_split(cfg.device)
    net = BPC(cfg)
    means_val, means_test = d.class_means("val"), d.class_means("test")
    xp, _ = d.probe
    g = torch.Generator().manual_seed(cfg.seed)
    n = len(d.y_train)
    hist, snaps = [], {}
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        perm = torch.randperm(n, generator=g).to(cfg.device)
        losses = []
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            losses.append(net.train_batch(d.x_train[idx], F.one_hot(d.y_train[idx], 10).float()))
        row = {"epoch": ep, "energy": float(np.mean(losses)), "time": round(time.time() - t0, 1)}
        if not np.isfinite(row["energy"]):
            row["diverged"] = True
            hist.append(row)
            break
        if ep % eval_every == 0 or ep == cfg.epochs or (save and ep in SNAPSHOTS):
            row["val_acc"] = accuracy(net, d.x_val, d.y_val)
            row["val_rmse"] = gen_rmse(net, means_val)[0]
        if save and ep in SNAPSHOTS:
            row["test_acc"] = accuracy(net, d.x_test, d.y_test)
            snaps[ep] = probe_states(net, xp)
        hist.append(row)
        if verbose:
            print(json.dumps(row), flush=True)
    rec = {"name": name, "config": cfg.to_dict(), "history": hist}
    if not hist[-1].get("diverged"):
        rec["test_acc"] = accuracy(net, d.x_test, d.y_test)
        rec["test_rmse"], img = gen_rmse(net, means_test)
        rec["val_acc"], rec["val_rmse"] = hist[-1]["val_acc"], hist[-1]["val_rmse"]
        rec["objective"] = 2 * (1 - rec["val_acc"]) + rec["val_rmse"]   # A.4 joint tuning objective
        if save:
            torch.save({"config": cfg.to_dict(), "state": net.state_dict()}, out / f"{name}.pt")
            np.savez_compressed(out / f"{name}__probe.npz", generated=img.cpu().numpy(),
                                **{f"e{ep}_{k}": v for ep, s in snaps.items() for k, v in s.items()})
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="pilot")
    p.add_argument("--save", action="store_true")
    p.add_argument("--eval_every", type=int, default=1)
    for k, v in BPCConfig().to_dict().items():
        if k == "sizes":
            continue
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = vars(p.parse_args())
    tag, save, ev = a.pop("tag"), a.pop("save"), a.pop("eval_every")
    rec = run(BPCConfig(**a), tag, save, ev)
    print(f"DONE {rec['name']}  test_acc={rec.get('test_acc')}  test_rmse={rec.get('test_rmse')}")


if __name__ == "__main__":
    main()
