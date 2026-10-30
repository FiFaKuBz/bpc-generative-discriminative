"""MNIST as in the bPC paper (A.1): pixels in [-1, 1]; the 10,000 test images split
50/50 into validation and test; plus a fixed class-balanced probe set for the
inside analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pcn.data import load

SPLIT_SEED = 2026
PROBE_SEED = 12345


@dataclass
class Split:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    probe_idx: np.ndarray          # indices into the test split, 100 per digit

    @property
    def probe(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x_test[self.probe_idx], self.y_test[self.probe_idx]

    def class_means(self, which: str = "test") -> torch.Tensor:
        x, y = (self.x_test, self.y_test) if which == "test" else (self.x_val, self.y_val)
        return torch.stack([x[y == c].mean(0) for c in range(10)])


def mnist_split(device: str = "cpu") -> Split:
    ds = load("mnist")
    to = lambda a: torch.from_numpy(a * 2.0 - 1.0).float().to(device)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(ds.y_test))
    iv, it = perm[:5000], perm[5000:]
    yt = ds.y_test[it]
    rng = np.random.default_rng(PROBE_SEED)
    probe = np.sort(np.concatenate([rng.choice(np.flatnonzero(yt == c), 100, replace=False) for c in range(10)]))
    lab = lambda a: torch.from_numpy(a.astype(np.int64)).to(device)
    return Split(to(ds.x_train), lab(ds.y_train), to(ds.x_test[iv]), lab(ds.y_test[iv]),
                 to(ds.x_test[it]), lab(yt), probe)
