"""discBP baseline of bPC §4.1: the same [784, 256, 256, 10] MLP trained by backprop
on the squared error to the one-hot label (Table 4), AdamW, 25 epochs, batch 256.
A small grid over the Table 5 range picks lr / weight decay on the validation split;
then 5 seeds are trained and their probe-set activations saved.

    PYTHONPATH=. ../PCN-Version2/.venv/bin/python scripts/train_bp.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcn.bpc import ACTS
from pcn.bpc_data import mnist_split

OUT = Path(__file__).resolve().parent.parent / "results" / "final"


class MLP(nn.Module):
    def __init__(self, act="leaky_relu", sizes=(784, 256, 256, 10)):
        super().__init__()
        self.lin = nn.ModuleList([nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])])
        self.f = ACTS[act]

    def forward(self, x):
        hs = []
        for i, lin in enumerate(self.lin):
            x = lin(x)
            if i < len(self.lin) - 1:
                x = self.f(x)
            hs.append(x)
        return hs                               # [L1, L2, L3(output)]


def train(seed, lr, wd, d, act="leaky_relu", epochs=25, bs=256):
    torch.manual_seed(seed)
    net = MLP(act)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    n = len(d.y_train)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = net(d.x_train[idx])[-1]
            loss = 0.5 * (out - F.one_hot(d.y_train[idx], 10).float()).pow(2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def acc(net, x, y):
    return float((net(x)[-1].argmax(1) == y).float().mean())


def main():
    torch.set_num_threads(4)
    d = mnist_split()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "bp_grid.json"
    grid = json.loads(cache.read_text()) if cache.exists() else []
    if not grid:
        for lr, wd in itertools.product((1e-3, 3e-4, 1e-4), (0.0, 1e-4, 1e-3)):
            a = acc(train(0, lr, wd, d), d.x_val, d.y_val)
            grid.append({"lr": lr, "wd": wd, "val_acc": a})
            print(grid[-1], flush=True)
        cache.write_text(json.dumps(grid, indent=1))
    best = max(grid, key=lambda r: r["val_acc"])
    xp, _ = d.probe
    recs = []
    for s in range(5):
        net = train(s, best["lr"], best["wd"], d)
        with torch.no_grad():
            hs = net(xp)
        np.savez_compressed(OUT / f"bp_s{s}__probe.npz", **{f"L{i + 1}": h.numpy().astype(np.float16) for i, h in enumerate(hs)})
        recs.append({"seed": s, "test_acc": acc(net, d.x_test, d.y_test)})
        print(recs[-1], flush=True)
    (OUT / "bp.json").write_text(json.dumps({"grid": grid, "best": best, "seeds": recs}, indent=1))


if __name__ == "__main__":
    main()
