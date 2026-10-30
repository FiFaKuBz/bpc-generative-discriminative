"""Data statistics for the slides: images per digit in each split (train / validation / test / probe).

Writes results/final/data_stats.json. Run:  PYTHONPATH=. python scripts/data_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pcn.bpc_data import mnist_split

OUT = Path(__file__).resolve().parents[1] / "results" / "final" / "data_stats.json"


def main() -> None:
    s = mnist_split()
    count = lambda y: np.bincount(np.asarray(y), minlength=10).tolist()
    stats = {
        "image_shape": [28, 28],
        "pixel_range": [float(s.x_train.min()), float(s.x_train.max())],
        "train": count(s.y_train), "val": count(s.y_val), "test": count(s.y_test),
        "probe": count(s.y_test[s.probe_idx]),
    }
    for k in ("train", "val", "test", "probe"):
        stats[f"n_{k}"] = int(sum(stats[k]))
    OUT.write_text(json.dumps(stats, indent=1))
    for k in ("train", "val", "test", "probe"):
        c = stats[k]
        print(f"{k:6s} n={sum(c):6d}  min {min(c)} (digit {int(np.argmin(c))})  max {max(c)} (digit {int(np.argmax(c))})  {c}")


if __name__ == "__main__":
    main()
