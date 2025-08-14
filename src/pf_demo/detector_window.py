"""
detector_window.py — Parameter-free window-stability particle detector
=====================================================================

Purpose
-------
A second detector that declares a cluster "particle" if, over a sliding window W,
its 3D centroid and radius are *stable* relative to the run-wide distribution
for that tick. Thresholds are *self-calibrating* (no fixed constants), derived
from robust statistics (MAD). The window length W is chosen automatically from
a global time series' autocorrelation (first local minimum), with a safe
fallback when insufficient data are present.

Intended use
------------
Run this detector *in parallel* with your periodic (loop-based) detector; a
cluster counts as a particle if EITHER detector says so. You can feed it the
same cluster IDs your periodic detector maintains (recommended for ID
persistence). If you don't have IDs, you may pass ephemeral IDs; this module
does not break (it just treats them as independent tracks).

Inputs per tick
---------------
- clusters: Dict[int, Set[int]]
    Mapping cluster_id -> set of node ids (visible nodes only).
- site: object with 'graph' and 'atlas' attributes
    'atlas.position(nid)' returns (x, y, z, tau); we use xyz only.

Optional:
- global_series_sample: float | None
    Any scalar time series that reflects global dynamics (e.g., mean memory
    density, total live clusters, sum of tags). If provided for ~100+ ticks,
    the detector will auto-pick W from its autocorrelation. If omitted,
    a conservative fallback W is used.

Outputs per tick
----------------
- stable_now: Dict[int, bool]
    Whether each cluster_id is currently stable under window-W test.
- stats_now: Dict[int, dict]
    Per-cluster snapshot (centroid, radius, var_radius, var_centroid, window).

No external knobs; safe, deterministic behavior.

Dependencies: numpy
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Small robust helpers
# --------------------------------------------------------------------------- #

def _mad_threshold(values: np.ndarray, k: float = 3.5) -> float:
    """
    Robust (median + k * MAD) threshold. No fixed physics—just a noise floor
    learned from the tick's own distribution.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("inf")
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    # 1.4826 scales MAD to std under normality; still robust for heavy tails.
    return float(med + k * 1.4826 * (mad + 1e-12))


def _first_local_minimum_acf(x: Sequence[float], max_lag: int = 200) -> Optional[int]:
    """
    Choose W = first local minimum of the normalized autocorrelation of x.
    Returns None if no reliable minimum is found.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return None
    x = (x - x.mean()) / (x.std() + 1e-12)

    # Compute one-sided autocorrelation (lags 0..max_lag or up to len-2)
    M = min(max_lag, len(x) - 2)
    if M < 4:
        return None

    ac = np.correlate(x, x, mode="full")[len(x) - 1 : len(x) - 1 + M + 1]
    ac = ac / (ac[0] + 1e-12)

    # First local minimum after lag >= 2
    for k in range(2, len(ac) - 1):
        if ac[k] < ac[k - 1] and ac[k] <= ac[k + 1]:
            return int(k)
    return None


def _centroid_and_radius(site, nodes: Iterable[int]) -> Tuple[np.ndarray, float]:
    """
    Compute 3D centroid and radius from atlas positions of given nodes.
    """
    pts = []
    for nid in nodes:
        # atlas.position returns (x,y,z,tau); we take xyz
        p = site.atlas.position(int(nid))
        pts.append((float(p[0]), float(p[1]), float(p[2])))
    if not pts:
        return np.zeros(3, dtype=float), 0.0
    P = np.asarray(pts, dtype=float)
    c = P.mean(axis=0)
    r = float(np.sqrt(((P - c) ** 2).sum(axis=1).mean()))
    return c, r


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class WindowStats:
    """Rolling statistics for one cluster."""
    centroids: deque   # of np.ndarray shape (3,)
    radii: deque       # of float

    def append(self, centroid: np.ndarray, radius: float, maxlen: int):
        if len(self.centroids) == self.centroids.maxlen:
            # deque auto-pops; just ensure dtype consistency
            pass
        self.centroids.append(centroid)
        self.radii.append(float(radius))

    def ready(self, w: int) -> bool:
        return len(self.radii) >= max(2, w)


# --------------------------------------------------------------------------- #
# Main detector
# --------------------------------------------------------------------------- #

class WindowStabilityDetector:
    """
    Parameter-free window-stability detector.

    Logic (per tick):
    1) (Optionally) update W using the global series' autocorrelation once we
       have enough samples; otherwise keep prior W or a conservative fallback.
    2) For each cluster, append (centroid, radius) to its history.
    3) If history length >= W, compute per-cluster:
          var_radius   = variance(radius over last W)
          var_centroid = variance(norm of centroid deltas over last W)
    4) Compute *distribution* of (var_radius) and (var_centroid) across all
       clusters that are ready; learn *tick-specific* robust thresholds via MAD.
    5) A cluster is stable if both its variances are <= learned thresholds.

    No fixed thresholds; learned anew each tick from actual data.
    """

    def __init__(
        self,
        site,
        max_history: int = 4096,
        fallback_window: int = 20,
        min_series_for_window: int = 64,
    ):
        self.site = site
        self.max_history = int(max_history)
        self.fallback_window = int(fallback_window)
        self.min_series_for_window = int(min_series_for_window)

        self._hist: Dict[int, WindowStats] = {}
        self._global_series: List[float] = []
        self._W: Optional[int] = None  # chosen window

        # cache: last-tick per-cluster stats for reporting
        self._last_stats: Dict[int, dict] = {}

    # ------------- public API -------------------------------------------- #

    @property
    def window(self) -> int:
        if self._W is None:
            return self.fallback_window
        return int(self._W)

    def feed_global_series(self, value: Optional[float]) -> None:
        """
        Add one sample to the global series used to choose W. You can pass:
        - mean memory density,
        - total live clusters,
        - or any scalar summary.
        """
        if value is None or not np.isfinite(value):
            return
        self._global_series.append(float(value))
        # Try to pick W once we have enough samples; do it only once or when W is stale
        if self._W is None and len(self._global_series) >= self.min_series_for_window:
            W = _first_local_minimum_acf(self._global_series, max_lag=200)
            # Keep it conservative if no minimum found
            self._W = W if (W is not None and W >= 4) else self.fallback_window

    def step(
        self,
        tick: int,
        clusters: Dict[int, Set[int]],
    ) -> Tuple[Dict[int, bool], Dict[int, dict]]:
        """
        Update histories with the current tick's clusters and decide stability.

        Parameters
        ----------
        tick : int
        clusters : dict[cluster_id -> set(node_id)]

        Returns
        -------
        stable_now : dict[int, bool]
        stats_now  : dict[int, dict]
        """
        # 1) Ensure histories exist and append new observations
        for cid, nodes in clusters.items():
            if cid not in self._hist:
                self._hist[cid] = WindowStats(
                    centroids=deque(maxlen=self.max_history),
                    radii=deque(maxlen=self.max_history),
                )
            c, r = _centroid_and_radius(self.site, nodes)
            self._hist[cid].append(c, r, self.max_history)

        # 2) Determine current window
        W = self.window

        # 3) Compute per-cluster window variances (for those with enough history)
        var_radius: Dict[int, float] = {}
        var_centroid: Dict[int, float] = {}

        ready_ids: List[int] = []
        for cid, H in self._hist.items():
            if not H.ready(W):
                continue
            ready_ids.append(cid)

            # radius variance over last W
            rW = np.asarray(list(H.radii)[-W:], dtype=float)
            var_radius[cid] = float(np.nanvar(rW))

            # centroid *step* magnitudes over last W (W-1 steps), then variance
            cW = np.asarray(list(H.centroids)[-W:], dtype=float)  # (W, 3)
            steps = np.linalg.norm(cW[1:] - cW[:-1], axis=1)
            var_centroid[cid] = float(np.nanvar(steps))

        # 4) Learn thresholds from the distribution at this tick
        if ready_ids:
            vr_all = np.array([var_radius[cid] for cid in ready_ids], dtype=float)
            vc_all = np.array([var_centroid[cid] for cid in ready_ids], dtype=float)
            thr_vr = _mad_threshold(vr_all, k=3.5)
            thr_vc = _mad_threshold(vc_all, k=3.5)
        else:
            thr_vr = thr_vc = float("inf")  # nothing ready → no one is stable

        # 5) Decide stability and prepare stats
        stable_now: Dict[int, bool] = {}
        stats_now: Dict[int, dict] = {}

        for cid, nodes in clusters.items():
            H = self._hist[cid]
            c, r = (H.centroids[-1], H.radii[-1]) if len(H.radii) else (np.zeros(3), 0.0)
            vr = var_radius.get(cid, float("inf"))
            vc = var_centroid.get(cid, float("inf"))
            is_stable = (vr <= thr_vr) and (vc <= thr_vc)

            stable_now[cid] = bool(is_stable)
            stats_now[cid] = {
                "tick": int(tick),
                "centroid": (float(c[0]), float(c[1]), float(c[2])),
                "radius": float(r),
                "var_radius_W": float(vr),
                "var_centroid_W": float(vc),
                "threshold_var_radius": float(thr_vr),
                "threshold_var_centroid": float(thr_vc),
                "window_W": int(W),
                "ready": bool(H.ready(W)),
            }

        self._last_stats = stats_now
        return stable_now, stats_now
