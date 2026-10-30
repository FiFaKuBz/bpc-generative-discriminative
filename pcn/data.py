"""Real MNIST / Fashion-MNIST loading, plus the prior-skew resampling used by Arm 2.

Nothing here is synthetic: the idx files are downloaded from the canonical
mirrors and parsed directly.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

MIRRORS = {
    "mnist": (
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        },
    ),
    "fashion": (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        },
    ),
}

# sha256 of the raw .gz files, recorded on first download so later runs can
# detect a corrupted or swapped cache.
CHECKSUM_FILE = "checksums.txt"


def _download(dataset: str, dest: Path) -> None:
    base, files = MIRRORS[dataset]
    dest.mkdir(parents=True, exist_ok=True)
    sums = {}
    for key, fname in files.items():
        target = dest / fname
        if not target.exists():
            url = base + fname
            print(f"  downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "pcn-poc/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                target.write_bytes(r.read())
        sums[fname] = hashlib.sha256(target.read_bytes()).hexdigest()
    (dest / CHECKSUM_FILE).write_text(
        "\n".join(f"{v}  {k}" for k, v in sorted(sums.items())) + "\n"
    )


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, ndim = struct.unpack(">HBB", f.read(4))[1:]
        shape = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(shape)


@dataclass
class Dataset:
    """Flattened, [0,1]-scaled images with integer labels."""

    x_train: np.ndarray  # (n, 784) float32
    y_train: np.ndarray  # (n,) int64
    x_test: np.ndarray
    y_test: np.ndarray
    name: str

    @property
    def n_classes(self) -> int:
        return int(self.y_train.max()) + 1

    @property
    def dim(self) -> int:
        return self.x_train.shape[1]


def load(dataset: str = "mnist") -> Dataset:
    if dataset not in MIRRORS:
        raise ValueError(f"unknown dataset {dataset!r}")
    dest = DATA_ROOT / dataset
    _, files = MIRRORS[dataset]
    if not all((dest / f).exists() for f in files.values()):
        print(f"[data] fetching real {dataset} into {dest}")
        _download(dataset, dest)

    xtr = _read_idx(dest / files["train_images"]).reshape(-1, 784)
    ytr = _read_idx(dest / files["train_labels"])
    xte = _read_idx(dest / files["test_images"]).reshape(-1, 784)
    yte = _read_idx(dest / files["test_labels"])

    return Dataset(
        x_train=(xtr.astype(np.float32) / 255.0),
        y_train=ytr.astype(np.int64),
        x_test=(xte.astype(np.float32) / 255.0),
        y_test=yte.astype(np.int64),
        name=dataset,
    )


# --------------------------------------------------------------------------
# Arm 2: the class prior, imposed as an empirical training frequency profile.
# --------------------------------------------------------------------------


def skew_weights(n_classes: int, skew: float, favoured: tuple[int, ...] | None = None) -> np.ndarray:
    """Relative sampling frequency per class.

    `skew` is the frequency ratio between favoured and disfavoured classes:
    skew=1.0 is a flat prior, skew=3.0 means favoured classes are seen 3x as
    often. Default favours the first half of the classes (digits 0-4).
    """
    if favoured is None:
        favoured = tuple(range(n_classes // 2))
    w = np.ones(n_classes, dtype=np.float64)
    w[list(favoured)] = skew
    return w / w.sum()


def resample_by_prior(
    y: np.ndarray,
    n_total: int,
    skew: float,
    rng: np.random.Generator,
    favoured: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Indices of a training set of size `n_total` whose class frequencies follow `skew`.

    Sampling is *without* replacement within a class where the class pool is
    large enough, so a skewed condition sees genuinely fewer distinct examples
    of the disfavoured classes -- that is the point: a weaker prior comes from
    less experience. Total training-set size is held constant across
    conditions (control C5), so only the *distribution* changes, never the
    training budget.
    """
    n_classes = int(y.max()) + 1
    w = skew_weights(n_classes, skew, favoured)
    per_class = np.floor(w * n_total).astype(int)
    # distribute the remainder to the largest-weight classes
    short = n_total - per_class.sum()
    if short > 0:
        per_class[np.argsort(-w)[:short]] += 1

    out = []
    for c in range(n_classes):
        pool = np.flatnonzero(y == c)
        k = per_class[c]
        if k <= pool.size:
            out.append(rng.choice(pool, size=k, replace=False))
        else:
            # need more than the pool holds -> take all, top up with replacement
            extra = rng.choice(pool, size=k - pool.size, replace=True)
            out.append(np.concatenate([pool, extra]))
    idx = np.concatenate(out)
    rng.shuffle(idx)
    return idx


def balanced_probe(
    y: np.ndarray, n_per_class: int, rng: np.random.Generator
) -> np.ndarray:
    """A fixed, class-balanced probe set (control C6).

    Every network in the study is measured on exactly these images, regardless
    of the prior it was trained under -- required for prediction P5.
    """
    n_classes = int(y.max()) + 1
    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(y == c), size=n_per_class, replace=False)
            for c in range(n_classes)
        ]
    )
    return np.sort(idx)


# --------------------------------------------------------------------------
# V2: the class prior as a pure FREQUENCY manipulation (fixes V1 Problem 3).
#
# V1 built a skewed training set by resampling: favoured digits were topped up
# with duplicates and disfavoured digits were cut to ~1,000 distinct images.
# So "seen less often" was confounded with "seen fewer different examples".
#
# V2 fixes the set of distinct images once -- the same pool for every network
# -- and changes only how often each digit is *presented*. Every image in the
# pool is still seen during training, whatever the skew. The "fewer distinct
# images" factor is then tested on its own by the diversity arm (balanced
# frequency, a smaller pool for digits 5-9).
# --------------------------------------------------------------------------


def fixed_pool(y: np.ndarray, per_class: int, seed: int = 2024) -> dict[int, np.ndarray]:
    """The same `per_class` distinct training images of every digit, for every
    network. 5,400 fits every MNIST digit (the rarest, '5', has 5,421)."""
    rng = np.random.default_rng(seed)
    return {c: np.sort(rng.choice(np.flatnonzero(y == c), size=per_class, replace=False))
            for c in range(int(y.max()) + 1)}


def restrict_pool(pool: dict[int, np.ndarray], classes, n_unique: int, seed: int = 7) -> dict[int, np.ndarray]:
    """Diversity arm: keep only `n_unique` distinct images for the given classes."""
    rng = np.random.default_rng(seed)
    out = dict(pool)
    for c in classes:
        out[c] = np.sort(rng.choice(pool[c], size=n_unique, replace=False))
    return out


class DietSampler:
    """Yields one epoch of `n_total` training indices at a time.

    Each class gets a fixed number of presentations per epoch (its weight x
    n_total, rounding slack to the heaviest classes). Within a class, images
    are drawn from a shuffled queue without replacement and the queue is
    refilled when empty, so every image of the pool is used before any is
    repeated. The training budget -- n_total presentations per epoch -- is the
    same for every condition (control C5).
    """

    def __init__(self, pool: dict[int, np.ndarray], weights: np.ndarray, n_total: int, seed: int):
        self.pool = pool
        self.n_total = n_total
        w = np.asarray(weights, dtype=np.float64) / np.sum(weights)
        per = np.floor(w * n_total).astype(int)
        short = n_total - per.sum()
        per[np.argsort(-w, kind="stable")[:short]] += 1
        self.per_class = per
        self.rng = np.random.default_rng(seed)
        self.queues = {c: np.empty(0, dtype=np.int64) for c in pool}

    def _take(self, c: int, k: int) -> np.ndarray:
        out = []
        while k > 0:
            if self.queues[c].size == 0:
                self.queues[c] = self.rng.permutation(self.pool[c])
            take = self.queues[c][:k]
            self.queues[c] = self.queues[c][k:]
            out.append(take)
            k -= take.size
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def epoch(self) -> np.ndarray:
        idx = np.concatenate([self._take(c, int(self.per_class[c])) for c in sorted(self.pool)])
        return self.rng.permutation(idx)
