"""
selfcal.py — Parameter-free, data-driven threshold selection
============================================================

This module provides *self-calibrating* choices for the thresholds that the
parameter-free demo needs. All selectors learn from the run’s own telemetry
(no hand-tuned constants beyond minimal safety clamps / fallbacks).

Provided functions
------------------
- choose_min_period(first_recurrence_samples, ...)
    → int  (minimal loop period to use this run)

- choose_jaccard_threshold(overlaps_0_to_1, ...)
    → float in (0,1)  (ID-persistence overlap threshold, learned via GMM)

- choose_min_size(noise_component_sizes, ...)
    → int  (size cutoff to ignore noise-sized “clusters”)

Design notes
------------
* Robust to outliers; works with small samples using conservative fallbacks.
* Deterministic (fixed random_state) where learning is involved.
* Safe clamps keep outputs in sensible ranges but do *not* inject physics.

Dependencies: numpy, scikit-learn (for GaussianMixture)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

try:
    # scikit-learn is already in your requirements.
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover
    GaussianMixture = None  # type: ignore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _as_1d_floats(x: Iterable[float]) -> np.ndarray:
    """Convert iterable to clean 1D float array, dropping NaN/inf."""
    arr = np.asarray(list(x), dtype=float).ravel()
    if arr.size == 0:
        return arr
    arr = arr[np.isfinite(arr)]
    return arr


def _robust_percentile(x: np.ndarray, p: float) -> float:
    """Percentile with guard for empty input."""
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, p))


def _logit_clip01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logit transform with clipping to avoid 0/1."""
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def _inv_logit(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SelfCalParams:
    """Optional convenience container if you want to bundle outputs."""
    min_loop_period: int
    jaccard_threshold: float
    min_particle_size: int


def choose_min_period(
    first_recurrence_samples: Iterable[float],
    pctl: float = 10.0,
    min_p: int = 3,
    max_p: int = 12,
    min_samples: int = 200,
) -> int:
    """
    Choose a *minimal* loop period for this run from empirical first-recurrence times.

    Heuristic: take the lower-decile (10th percentile) of first recurrences, which
    discards the quickest “flicker” loops but keeps genuine cyclicity. Clamp to
    [min_p, max_p] for stability and require a minimal sample size before trusting it.

    Fallback: returns min_p if not enough data.

    Parameters
    ----------
    first_recurrence_samples : iterable of float
        Per-node first recurrence (in ticks) gathered during warmup.
    pctl : float
        Percentile to use (default 10th).
    min_p, max_p : int
        Safety bounds on the chosen period.
    min_samples : int
        Minimum number of samples required to trust the estimate.

    Returns
    -------
    int
    """
    x = _as_1d_floats(first_recurrence_samples)
    if x.size < min_samples:
        return int(min_p)
    est = _robust_percentile(x, pctl)
    if not np.isfinite(est):
        return int(min_p)
    return int(np.clip(round(est), min_p, max_p))


def choose_jaccard_threshold(
    overlaps_0_to_1: Iterable[float],
    clamp: Tuple[float, float] = (0.2, 0.8),
    min_pairs: int = 100,
    random_state: int = 0,
) -> float:
    """
    Learn an ID-persistence (Jaccard) threshold from the data.

    Model: fit a 2-component Gaussian Mixture to the *logit* of candidate
    overlaps (same-cluster vs different-cluster). Use the *equal-likelihood*
    intersection point of the two Gaussians as the operating threshold.

    This adapts automatically across runs: when overlaps are bimodal, the
    crossing lies between the modes; when unimodal or too few pairs exist,
    we fall back to a conservative value near 0.5.

    Parameters
    ----------
    overlaps_0_to_1 : iterable of float
        Candidate Jaccard overlaps in [0,1].
    clamp : (low, high)
        Clamp the final threshold to this interval for safety.
    min_pairs : int
        Require at least this many valid pairs before trusting GMM.
    random_state : int
        For deterministic GMM fitting.

    Returns
    -------
    float in (0,1)
    """
    arr = _as_1d_floats(overlaps_0_to_1)
    # keep strict (0,1); drop degenerate 0/1 which carry little info
    arr = arr[(arr > 0.0) & (arr < 1.0)]
    if arr.size < min_pairs or GaussianMixture is None:
        # fallback: robust central tendency (median) clamped to [0.2,0.8]
        med = float(np.median(arr)) if arr.size else 0.5
        return float(np.clip(med, *clamp))

    z = _logit_clip01(arr)
    # Fit 2-component GMM in logit space
    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
    gm.fit(z.reshape(-1, 1))
    means = gm.means_.ravel()
    vars_ = gm.covariances_.ravel()
    stds = np.sqrt(vars_)

    # Order components by mean
    order = np.argsort(means)
    m1, m2 = float(means[order[0]]), float(means[order[1]])
    s1, s2 = float(stds[order[0]]), float(stds[order[1]])
    w1, w2 = float(gm.weights_[order[0]]), float(gm.weights_[order[1]])

    # Solve for z where w1*N(z|m1,s1) = w2*N(z|m2,s2)
    # This reduces to a quadratic: a z^2 + b z + c = 0
    # Derivation uses log equality of weighted Gaussians.
    eps = 1e-12
    a = 0.5 * (1.0 / (s1**2 + eps) - 1.0 / (s2**2 + eps))
    b = (m2 / (s2**2 + eps)) - (m1 / (s1**2 + eps))
    c = 0.5 * ((m1**2) / (s1**2 + eps) - (m2**2) / (s2**2 + eps)) + np.log(
        (w2 * (s1 + eps)) / (w1 * (s2 + eps) + eps)
    )

    z_star: float
    if abs(a) < 1e-12:
        # Nearly equal variances -> linear crossing
        if abs(b) < 1e-12:
            z_star = 0.5 * (m1 + m2)  # fallback midway
        else:
            z_star = -c / b
    else:
        roots = np.roots([a, b, c])  # type: ignore[arg-type]
        # pick the real root that lies between the means (if any)
        real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
        between = [r for r in real_roots if min(m1, m2) <= r <= max(m1, m2)]
        if between:
            z_star = between[0]
        elif real_roots:
            # pick the closest to the interval if none inside
            z_star = min(real_roots, key=lambda r: min(abs(r - m1), abs(r - m2)))
        else:
            z_star = 0.5 * (m1 + m2)

    t = _inv_logit(z_star)
    return float(np.clip(t, *clamp))


def choose_min_size(
    noise_component_sizes: Iterable[float],
    pctl: float = 99.0,
    min_size: int = 2,
    max_size: int | None = None,
    min_samples: int = 200,
) -> int:
    """
    Choose a size cutoff that filters out noise-like components.

    Heuristic: estimate the 99th percentile of *noise* connected-component sizes
    (measured among non-periodic nodes). Anything smaller than or equal to this
    cutoff is treated as noise; particles must be strictly larger.

    Fallback: conservative default of max(min_size, 3) if data are scarce.

    Parameters
    ----------
    noise_component_sizes : iterable of float
        Sizes of connected components that are *not* flagged periodic.
    pctl : float
        Upper percentile used as a noise floor (default 99%).
    min_size : int
        Lower bound for safety.
    max_size : Optional[int]
        Optional upper bound to guard against pathological spikes.
    min_samples : int
        Minimum number of components to trust the estimate.

    Returns
    -------
    int
    """
    x = _as_1d_floats(noise_component_sizes)
    if x.size < min_samples:
        return max(min_size, 3)
    est = _robust_percentile(x, pctl)
    if not np.isfinite(est):
        return max(min_size, 3)
    val = int(max(min_size, np.floor(est) + 1))  # strictly above noise floor
    if max_size is not None:
        val = int(min(val, max_size))
    return val
