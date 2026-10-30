"""Representational divergence and compression measures.

Control C1: every comparison here is invariant to permutation of hidden units,
rotation of the basis, and isotropic rescaling -- the symmetries under which a
PCN's computation is unchanged. Comparing raw activations would measure those
symmetries instead of anything real, which is the one fatal mistake in this
design.
"""

from __future__ import annotations

import numpy as np
import torch


def _center(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    return x - x.mean(0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) on centered matrices.

    X, Y are (n_probe, d) -- d may differ between them.
    Invariant to orthogonal transforms and isotropic scaling.
    """
    X, Y = _center(X), _center(Y)
    xty = Y.T @ X
    num = float((xty ** 2).sum())
    den = float(torch.linalg.matrix_norm(X.T @ X) * torch.linalg.matrix_norm(Y.T @ Y))
    if den <= 0:
        return float("nan")
    return num / den


def cka_divergence(X: torch.Tensor, Y: torch.Tensor) -> float:
    return 1.0 - linear_cka(X, Y)


def procrustes_divergence(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Orthogonal-Procrustes distance after centering and Frobenius normalisation.

    d^2 = ||X||^2 + ||Y||^2 - 2*||X^T Y||_*  with both scaled to unit norm,
    so d in [0, sqrt(2)]; returned normalised to [0, 1]. This is the robustness
    check against CKA: they weight the spectrum differently, so agreement
    between them is evidence the effect is not a metric artifact.
    """
    X, Y = _center(X), _center(Y)
    X = X / (torch.linalg.matrix_norm(X) + 1e-12)
    Y = Y / (torch.linalg.matrix_norm(Y) + 1e-12)
    nuc = float(torch.linalg.svdvals(X.T @ Y).sum())
    d2 = max(0.0, 2.0 - 2.0 * nuc)
    return float(np.sqrt(d2) / np.sqrt(2.0))


def participation_ratio(X: torch.Tensor) -> float:
    """Effective dimensionality: (sum lam)^2 / sum lam^2 of the representation covariance.

    A compression measure that needs no density estimation -- unlike a direct
    I(X;T) estimate, whose binning dependence is exactly what the
    Tishby/Saxe dispute turned on.
    """
    Xc = _center(X)
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(1, n - 1)
    lam = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1 = float(lam.sum())
    s2 = float((lam ** 2).sum())
    if s2 <= 0:
        return float("nan")
    return (s1 ** 2) / s2


def diag_laplace_kl(
    X: torch.Tensor, hess_diag: torch.Tensor, prior_var: float = 1.0
) -> float:
    """KL( N(x*, diag(1/h)) || N(0, prior_var I) ) in nats, per sample.

    A Laplace approximation around the PCN's MAP equilibrium: the posterior
    precision is the Hessian of F, approximated by its diagonal. Reported as a
    *secondary*, explicitly approximate rate measure alongside the exact
    free-energy complexity term.
    """
    h = hess_diag.to(torch.float64).clamp(min=1e-8)
    mu = X.to(torch.float64)
    var = 1.0 / h
    d = mu.shape[1]
    kl = 0.5 * (
        (var / prior_var).sum(1)
        + (mu ** 2).sum(1) / prior_var
        - d
        - torch.log(var / prior_var).sum(1)
    )
    return float(kl.mean())


def layer_summary(X: torch.Tensor) -> dict:
    """Compression descriptors for one layer's probe-set representation."""
    Xc = _center(X)
    n, d = Xc.shape
    cov = (Xc.T @ Xc) / max(1, n - 1)
    lam = torch.linalg.eigvalsh(cov).clamp(min=0).flip(0)
    s1 = float(lam.sum())
    s2 = float((lam ** 2).sum())
    pr = (s1 ** 2) / s2 if s2 > 0 else float("nan")
    csum = torch.cumsum(lam, 0) / (s1 + 1e-12)
    return {
        "participation_ratio": pr,
        "dim": int(d),
        "total_variance": s1,
        "dim_90pct": int((csum < 0.90).sum()) + 1,   # components for 90% variance
        "dim_99pct": int((csum < 0.99).sum()) + 1,
        "mean_abs_activation": float(X.abs().mean()),
    }


# --------------------------------------------------------------------------
# pairwise divergence bookkeeping
# --------------------------------------------------------------------------


def pairwise_divergence(
    reps_a: list[torch.Tensor],
    reps_b: list[torch.Tensor],
    layers: list[int],
    metric: str = "cka",
) -> dict[int, float]:
    """Divergence between two networks' representations, per layer."""
    fn = cka_divergence if metric == "cka" else procrustes_divergence
    return {l: fn(reps_a[l], reps_b[l]) for l in layers}


def class_wise_divergence(
    reps_a: list[torch.Tensor],
    reps_b: list[torch.Tensor],
    probe_labels: np.ndarray,
    layers: list[int],
    metric: str = "cka",
) -> dict[int, dict[int, float]]:
    """Divergence computed *within each class* separately (prediction P5).

    Seed noise should be uniform across classes; a genuine prior effect should
    concentrate on the classes whose priors differ.
    """
    fn = cka_divergence if metric == "cka" else procrustes_divergence
    out: dict[int, dict[int, float]] = {}
    classes = np.unique(probe_labels)
    for l in layers:
        per_class = {}
        for c in classes:
            m = np.flatnonzero(probe_labels == c)
            if m.size < 8:
                continue
            idx = torch.from_numpy(m)
            per_class[int(c)] = fn(reps_a[l][idx], reps_b[l][idx])
        out[l] = per_class
    return out
