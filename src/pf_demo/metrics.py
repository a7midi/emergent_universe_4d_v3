"""
metrics.py — Core statistics for the parameter-free demo
=======================================================

What this module provides
-------------------------
• ks_distance(a, b)
    Two-sample Kolmogorov–Smirnov distance with robust NaN filtering.

• hill_tail_index(x, k_frac=0.10, min_k=20, max_k=None)
    Hill estimator for the tail exponent of positive heavy-tailed data
    (e.g., lifetimes). Returns (alpha_hat, k_used).

• spearman_rank_corr(x, y)
    Spearman rho with NaN-safe alignment.

• ecdf(x), ccdf(x)
    Empirical CDF / complementary CDF (returns (xs, F(xs))).

• normalize_by_median(x), normalize_by_quantile(x, q=0.5)
    Dimensionless normalization helpers.

• ks_grid(samples)
    Pairwise KS matrix and its max; useful for “invariance across runs.”

• ks_between_levels(level_to_samples)
    KS distances between RG level 0 and others; returns dict[level] -> KS.

• histogram_density(x, bins='fd')
    (bin_edges, density) with probability-mass normalization.

All functions are pure and side-effect free. Dependencies: numpy, scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import ks_2samp, spearmanr
except Exception:  # pragma: no cover
    ks_2samp = None  # type: ignore
    spearmanr = None  # type: ignore


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _clean_1d(x: Iterable[float]) -> np.ndarray:
    """1D float array, drop NaN/inf."""
    a = np.asarray(list(x), dtype=float).ravel()
    if a.size == 0:
        return a
    a = a[np.isfinite(a)]
    return a


def _require_pos(x: np.ndarray) -> np.ndarray:
    """Keep strictly positive values (for tail estimation)."""
    return x[x > 0.0]


# --------------------------------------------------------------------------- #
# KS distance and grids
# --------------------------------------------------------------------------- #

def ks_distance(a: Iterable[float], b: Iterable[float]) -> float:
    """
    Two-sample Kolmogorov–Smirnov distance D in [0,1].
    Returns 0.0 if either sample lacks enough points.
    """
    aa = _clean_1d(a)
    bb = _clean_1d(b)
    if aa.size < 2 or bb.size < 2 or ks_2samp is None:
        return 0.0
    return float(ks_2samp(aa, bb, alternative="two-sided", mode="auto").statistic)


def ks_grid(samples: Sequence[Iterable[float]]) -> Tuple[np.ndarray, float]:
    """
    Pairwise KS distance matrix for a sequence of samples.
    Returns (D, D_max). If any sample is tiny, its row/col will be zeros.
    """
    n = len(samples)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dij = ks_distance(samples[i], samples[j])
            D[i, j] = D[j, i] = dij
    return D, float(np.max(D) if D.size else 0.0)


def ks_between_levels(level_to_samples: Mapping[int, Iterable[float]]) -> Dict[int, float]:
    """
    Compare RG level 0 to other levels via KS distance.
    Returns dict[level] -> KS( level0 , level ).
    Missing or tiny samples yield 0.0.
    """
    if 0 not in level_to_samples:
        return {lvl: 0.0 for lvl in level_to_samples.keys()}
    ks: Dict[int, float] = {}
    base = level_to_samples[0]
    for lvl, samp in level_to_samples.items():
        if lvl == 0:
            continue
        ks[lvl] = ks_distance(base, samp)
    return ks


# --------------------------------------------------------------------------- #
# Rank correlation
# --------------------------------------------------------------------------- #

def spearman_rank_corr(x: Iterable[float], y: Iterable[float]) -> float:
    """
    Spearman rho in [-1,1], with NaN-/inf-safe alignment. Returns 0.0 if too small.
    """
    X = _clean_1d(x)
    Y = _clean_1d(y)
    n = min(X.size, Y.size)
    if n < 3 or spearmanr is None:
        return 0.0
    X = X[:n]
    Y = Y[:n]
    r = spearmanr(X, Y, nan_policy="omit")
    # scipy returns object with .correlation; older versions may return tuple
    rho = getattr(r, "correlation", r[0] if isinstance(r, tuple) else r)
    if not np.isfinite(rho):
        return 0.0
    return float(rho)


# --------------------------------------------------------------------------- #
# Empirical CDF/CCDF and normalization
# --------------------------------------------------------------------------- #

def ecdf(x: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical CDF: returns (xs, F(xs)), with xs sorted unique.
    """
    a = _clean_1d(x)
    if a.size == 0:
        return np.zeros(0), np.zeros(0)
    xs = np.sort(a)
    F = np.arange(1, xs.size + 1, dtype=float) / xs.size
    return xs, F


def ccdf(x: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complementary CDF (1 - CDF). Useful for heavy-tail inspection.
    """
    xs, F = ecdf(x)
    return xs, (1.0 - F)


def normalize_by_median(x: Iterable[float]) -> np.ndarray:
    """
    Divide by the sample median (dimensionless). If median<=0, return original.
    """
    a = _clean_1d(x)
    if a.size == 0:
        return a
    m = float(np.median(a))
    if not np.isfinite(m) or m <= 0.0:
        return a
    return a / m


def normalize_by_quantile(x: Iterable[float], q: float = 0.5) -> np.ndarray:
    """
    Divide by the q-quantile (dimensionless). If q<=0 value, return original.
    """
    a = _clean_1d(x)
    if a.size == 0:
        return a
    m = float(np.quantile(a, q))
    if not np.isfinite(m) or m <= 0.0:
        return a
    return a / m


def histogram_density(x: Iterable[float], bins: int | str = "fd") -> Tuple[np.ndarray, np.ndarray]:
    """
    Probability-mass histogram: returns (bin_edges, density) with sum(density * bin_width) = 1.
    """
    a = _clean_1d(x)
    if a.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0])
    dens, edges = np.histogram(a, bins=bins, density=True)
    # Ensure finite
    dens = np.nan_to_num(dens, nan=0.0, posinf=0.0, neginf=0.0)
    return edges.astype(float), dens.astype(float)


# --------------------------------------------------------------------------- #
# Hill estimator (tail index)
# --------------------------------------------------------------------------- #

def hill_tail_index(
    x: Iterable[float],
    k_frac: float = 0.10,
    min_k: int = 20,
    max_k: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Hill estimator for positive data's tail exponent alpha (>0) using the top-k order stats.

    For heavy-tailed lifetimes L with P(L > t) ~ t^{-alpha}, the Hill estimator on
    the top-k samples x_(n-k+1) ... x_(n) is:

        1/alpha_hat = (1/k) * sum_{i=1..k} [ log x_(n+1-i) - log x_(n-k) ]

    Returns (alpha_hat, k_used). If data insufficient or degenerate, returns (nan, 0).

    Parameters
    ----------
    x : iterable of float
        Sample (must be positive values).
    k_frac : float
        Fraction of the sample to use as the tail (0 < k_frac < 1).
    min_k : int
        Minimum k for stability.
    max_k : Optional[int]
        Cap on k; if None, use floor(k_frac * n).
    """
    a = _require_pos(_clean_1d(x))
    n = a.size
    if n < max(min_k, 5):
        return float("nan"), 0

    a.sort()
    # choose k
    k = int(np.floor(k_frac * n))
    if max_k is not None:
        k = min(k, int(max_k))
    k = max(k, int(min_k))
    if k >= n:
        k = n - 1
    if k <= 0:
        return float("nan"), 0

    x_tail = a[-k:]
    x_kth = a[-k - 1] if (n - k - 1) >= 0 else a[0]
    if x_kth <= 0.0 or np.any(x_tail <= 0.0):
        return float("nan"), 0

    logs = np.log(x_tail) - np.log(x_kth)
    inv_alpha = float(np.mean(logs))
    if inv_alpha <= 0.0 or not np.isfinite(inv_alpha):
        return float("nan"), 0
    alpha_hat = 1.0 / inv_alpha
    return float(alpha_hat), int(k)


# --------------------------------------------------------------------------- #
# Bundles for convenience (optional)
# --------------------------------------------------------------------------- #

@dataclass
class DistSummary:
    """
    Bundle a distribution and normalized variant for convenience.
    """
    raw: np.ndarray
    norm_median: np.ndarray

    @classmethod
    def from_data(cls, x: Iterable[float]) -> "DistSummary":
        arr = _clean_1d(x)
        return cls(raw=arr, norm_median=normalize_by_median(arr))


@dataclass
class InvarianceReport:
    """
    Simple container for invariance checks across runs.
    """
    ks_period_norm_max: float
    ks_speed_norm_max: float
    ks_size_norm_max: float
    hill_alpha_mean: float
    hill_alpha_std: float
    atlas_graph_spearman: float


def summarize_invariance(
    period_samples: Sequence[Iterable[float]],
    speed_samples: Sequence[Iterable[float]],
    size_samples: Sequence[Iterable[float]],
    lifetime_samples: Sequence[Iterable[float]],
    atlas_speeds_all: Iterable[float],
    graph_speeds_all: Iterable[float],
) -> InvarianceReport:
    """
    Compute the headline metrics used by the PASS/MIXED/FAIL logic.
    Input samples should be *already filtered* for each run (one entry per run).
    """
    # KS grids on median-normalized data
    P = [normalize_by_median(p) for p in period_samples]
    S = [normalize_by_median(s) for s in speed_samples]
    Z = [normalize_by_median(z) for z in size_samples]

    _, ksP = ks_grid(P)
    _, ksS = ks_grid(S)
    _, ksZ = ks_grid(Z)

    # Hill tail index per run
    alphas = []
    for lif in lifetime_samples:
        a, k = hill_tail_index(lif)
        if np.isfinite(a):
            alphas.append(a)
    hill_mean = float(np.mean(alphas)) if alphas else float("nan")
    hill_std  = float(np.std(alphas)) if alphas else float("nan")

    # Atlas vs graph speed correlation
    rho = spearman_rank_corr(atlas_speeds_all, graph_speeds_all)

    return InvarianceReport(
        ks_period_norm_max=float(ksP),
        ks_speed_norm_max=float(ksS),
        ks_size_norm_max=float(ksZ),
        hill_alpha_mean=hill_mean,
        hill_alpha_std=hill_std,
        atlas_graph_spearman=float(rho),
    )
