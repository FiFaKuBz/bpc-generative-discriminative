"""The four pre-registered tests of section 6, in ascending persuasiveness.

Every effect is judged against the seed-only floor, never against zero.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class EffectResult:
    n_manip: int
    n_seed: int
    mean_manip: float
    mean_seed: float
    sd_manip: float
    sd_seed: float
    cohens_d: float
    auc: float
    perm_p: float
    fraction_of_ceiling: float | None = None
    ceiling: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def verdict(self, d_thr=0.8, auc_thr=0.8, p_thr=0.01) -> bool:
        return (
            self.cohens_d > d_thr and self.auc > auc_thr and self.perm_p < p_thr
        )


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised difference, pooled SD."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(sp2))


def auc_superiority(a: np.ndarray, b: np.ndarray) -> float:
    """P(a random draw from `a` exceeds a random draw from `b`), ties at 0.5.

    Assumption-free overlap measure (the common-language effect size).
    Computed exactly via rank statistics.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_a = ranks[:na].sum()
    u_a = r_a - na * (na + 1) / 2.0
    return float(u_a / (na * nb))


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """One-sided p for mean(a) > mean(b) by label shuffling. No distributional assumptions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if pool[:na].mean() - pool[na:].mean() >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def compare_to_floor(
    manip: np.ndarray,
    seed_floor: np.ndarray,
    ceiling: float | None = None,
    n_perm: int = 10000,
    seed: int = 0,
) -> EffectResult:
    manip = np.asarray(manip, float)
    seed_floor = np.asarray(seed_floor, float)
    frac = None
    if ceiling is not None and np.isfinite(ceiling):
        rng = ceiling - seed_floor.mean()
        frac = float((manip.mean() - seed_floor.mean()) / rng) if rng > 0 else None
    return EffectResult(
        n_manip=len(manip),
        n_seed=len(seed_floor),
        mean_manip=float(manip.mean()),
        mean_seed=float(seed_floor.mean()),
        sd_manip=float(manip.std(ddof=1)) if len(manip) > 1 else float("nan"),
        sd_seed=float(seed_floor.std(ddof=1)) if len(seed_floor) > 1 else float("nan"),
        cohens_d=cohens_d(manip, seed_floor),
        auc=auc_superiority(manip, seed_floor),
        perm_p=permutation_test(manip, seed_floor, n_perm, seed),
        ceiling=float(ceiling) if ceiling is not None else None,
        fraction_of_ceiling=frac,
    )


# --------------------------------------------------------------------------
# dose-response and accuracy control (C3)
# --------------------------------------------------------------------------


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple least squares with slope SE, t and R^2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3 or x.std() == 0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "se": float("nan"), "t": float("nan"), "n": n}
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    dof = n - 2
    s2 = (resid ** 2).sum() / dof
    se = np.sqrt(s2 / sxx)
    sst = ((y - ym) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / sst if sst > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2),
            "se": float(se), "t": float(slope / se) if se > 0 else float("nan"), "n": n}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation -- monotonicity of the dose-response without assuming linearity."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")

    def rank(v):
        o = v.argsort()
        r = np.empty(len(v), float)
        r[o] = np.arange(1, len(v) + 1)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        return (s / cnt)[inv]

    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def partial_effect_controlling_accuracy(
    divergence: np.ndarray, acc_gap: np.ndarray, is_manip: np.ndarray
) -> dict:
    """Control C3: regress divergence on |accuracy gap| AND condition.

    Returns the condition coefficient -- the divergence attributable to the
    manipulation *after* removing anything explained by the pair's accuracy
    difference. Blocks the "you just compared a better net to a worse one"
    objection.
    """
    y = np.asarray(divergence, float)
    X = np.column_stack(
        [np.ones(len(y)), np.asarray(acc_gap, float), np.asarray(is_manip, float)]
    )
    if len(y) < 4:
        return {"beta_condition": float("nan"), "beta_accgap": float("nan"),
                "t_condition": float("nan"), "n": len(y)}
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    s2 = (resid ** 2).sum() / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(s2 * xtx_inv))
    return {
        "beta_condition": float(beta[2]),
        "beta_accgap": float(beta[1]),
        "se_condition": float(se[2]),
        "t_condition": float(beta[2] / se[2]) if se[2] > 0 else float("nan"),
        "n": int(len(y)),
    }


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
