from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.utils.graph_algorithms import find_connected_clusters
from src.kinematics import calculate_kinematics


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _path_sig(tag: int, preds: List[bytes]) -> bytes:
    """Order-insensitive hash of (node tag, predecessor signatures)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(int(tag).to_bytes(4, "little"))
    for s in sorted(preds):
        h.update(s)
    return h.digest()


def _proper_divisors(n: int) -> Iterable[int]:
    """Return all proper divisors of n in increasing order (excluding n)."""
    if n <= 1:
        return []
    small, large = [], []
    d = 1
    while d * d <= n:
        if n % d == 0:
            if d != n:
                small.append(d)
            q = n // d
            if q != d and q != n:
                large.append(q)
        d += 1
    return sorted(small + large)


def _is_period(history: deque[bytes], p: int) -> bool:
    """
    Check if the tail of the history is p-periodic over two consecutive blocks:
      ... x[-2p:-p] == x[-p:].
    Requires length >= 2p to be reliable.
    """
    if p <= 0 or len(history) < 2 * p:
        return False
    for i in range(1, p + 1):
        if history[-i] != history[-i - p]:
            return False
    return True


def _confirm_min_period(history: deque[bytes], candidate_p: int, min_period: int) -> Optional[int]:
    """
    Given a candidate period p (distance to last equal signature), confirm the *minimal*
    period ≥ min_period by checking proper divisors first, then p itself. Returns None
    if we cannot confirm periodicity on the available history.
    """
    if candidate_p < min_period:
        return None
    for d in _proper_divisors(candidate_p):
        if d >= min_period and _is_period(history, d):
            return d
    return candidate_p if _is_period(history, candidate_p) else None


def _jaccard(a: FrozenSet[int], b: FrozenSet[int]) -> float:
    """Jaccard overlap |A∩B| / |A∪B| (returns 0.0 if both empty)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Particle:
    id: int
    period: int
    nodes: FrozenSet[int]
    first_tick: int
    last_tick: int
    kinematics: dict

    @property
    def lifetime(self) -> int:
        return self.last_tick - self.first_tick


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class ParticleDetector:
    """
    Detects periodic, indecomposable clusters and tracks them through time.

    Steps each tick:
      1) Build per-node path signatures (hash of own tag + predecessor signatures).
      2) Detect loops via signature recurrences and confirm *minimal* period.
      3) Cluster looping nodes by graph connectivity.
      4) Greedy-match current clusters to last-tick clusters by Jaccard overlap, so
         particle IDs persist across small deformations and motion.
      5) Update kinematics with tick-time velocities.

    Config keys (detector section):
      - max_history_length (int)
      - min_loop_period (int)
      - min_particle_size (int)
      - match_jaccard_threshold (float in [0,1], default 0.5)
    """

    def __init__(self, site, state_mgr, cfg):
        self.site = site
        self.state_mgr = state_mgr

        d = cfg.get("detector", {})
        self.max_hist: int = int(d.get("max_history_length", 10000))
        self.min_period: int = int(d.get("min_loop_period", 3))
        self.min_size: int = int(d.get("min_particle_size", 2))
        self.jaccard_tau: float = float(d.get("match_jaccard_threshold", 0.5))

        # Per-node signature history and current signature cache
        self._hist: Dict[int, deque[bytes]] = {
            int(n): deque(maxlen=self.max_hist) for n in self.site.graph.nodes
        }
        self._sig_cache: Dict[int, bytes] = {}

        # Particle stores
        self._next_pid: int = 0
        self.active: Dict[int, Particle] = {}
        self.archive: Dict[int, Particle] = {}

        # Telemetry
        self.looping_nodes_last_tick: Set[int] = set()

    # ------------------------------------------------------------------- #
    def _visible_nodes(self, state: np.ndarray) -> np.ndarray:
        nodes = np.arange(state.shape[0])
        hidden = self.state_mgr.hidden_nodes
        if not hidden:
            return nodes
        mask = np.ones_like(nodes, dtype=bool)
        mask[list(hidden)] = False
        return nodes[mask]

    # ------------------------------------------------------------------- #
    def _current_clusters(self, state: np.ndarray, tick: int) -> List[Tuple[int, FrozenSet[int]]]:
        """
        Return list of (period, frozenset(node_ids)) for clusters detected this tick.
        """
        vis = self._visible_nodes(state)

        # 1) Update path signatures for visible nodes
        cur: Dict[int, bytes] = {}
        for n in vis:
            preds = [self._sig_cache.get(p, b"\0" * 16) for p in self.site.get_predecessors(int(n))]
            cur[int(n)] = _path_sig(int(state[int(n)]), preds)
        self._sig_cache = cur

        # 2) Detect loops and confirm minimal period
        loops: Dict[int, Set[int]] = {}
        for n in vis:
            n = int(n)
            sig = cur[n]
            h = self._hist[n]
            if sig in h:
                candidate_p = len(h) - 1 - h.index(sig)
                p = _confirm_min_period(h, candidate_p, self.min_period)
                if p is not None:
                    loops.setdefault(p, set()).add(n)
            h.append(sig)

        self.looping_nodes_last_tick = {m for s in loops.values() for m in s}

        # 3) Cluster looping nodes by connectivity
        clusters: List[Tuple[int, FrozenSet[int]]] = []
        for period, nodes_set in loops.items():
            for clust in find_connected_clusters(nodes_set, self.site):
                if len(clust) >= self.min_size:
                    clusters.append((int(period), frozenset(int(x) for x in clust)))

        return clusters

    # ------------------------------------------------------------------- #
    def _greedy_match_by_jaccard(
        self,
        prev: List[Tuple[int, FrozenSet[int], int]],  # (period, nodes, pid)
        curr: List[Tuple[int, FrozenSet[int]]],       # (period, nodes)
    ) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
        """
        Greedy 1-1 matching of prev clusters (with pids) to current clusters by Jaccard.
        Returns:
          matches: list of (prev_idx, curr_idx)
          prev_unmatched: set of prev indices
          curr_unmatched: set of curr indices
        """
        if not prev or not curr:
            return [], set(range(len(prev))), set(range(len(curr)))

        # Build all candidate pairs above threshold
        cand: List[Tuple[float, int, int]] = []  # (score, i_prev, j_curr)
        for i, (_, a_nodes, _) in enumerate(prev):
            for j, (_, b_nodes) in enumerate(curr):
                s = _jaccard(a_nodes, b_nodes)
                if s >= self.jaccard_tau:
                    cand.append((s, i, j))
        cand.sort(reverse=True)  # greedy highest-overlap first

        matched_prev: Set[int] = set()
        matched_curr: Set[int] = set()
        matches: List[Tuple[int, int]] = []
        for s, i, j in cand:
            if i in matched_prev or j in matched_curr:
                continue
            matched_prev.add(i)
            matched_curr.add(j)
            matches.append((i, j))

        prev_unmatched = set(range(len(prev))) - matched_prev
        curr_unmatched = set(range(len(curr))) - matched_curr
        return matches, prev_unmatched, curr_unmatched

    # ------------------------------------------------------------------- #
    def detect(self, state: np.ndarray, tick: int) -> Dict[int, Particle]:
        """
        Update detector with the current state and return the dict of active particles.
        """
        # Build current clusters
        curr_clusters = self._current_clusters(state, tick)  # List[(period, nodes)]
        # Snapshot previous active clusters for matching
        prev_clusters: List[Tuple[int, FrozenSet[int], int]] = [
            (p.period, p.nodes, pid) for pid, p in self.active.items()
        ]

        # Greedy Jaccard matching
        matches, prev_unmatched, curr_unmatched = self._greedy_match_by_jaccard(prev_clusters, curr_clusters)

        # Update matched particles in-place
        for i_prev, j_curr in matches:
            prev_period, prev_nodes, pid = prev_clusters[i_prev]
            cur_period, cur_nodes = curr_clusters[j_curr]
            last_kin = self.active[pid].kinematics

            kin = calculate_kinematics(
                Particle(pid, cur_period, cur_nodes, self.active[pid].first_tick, tick, {}),
                self.site.atlas,
                self.site.metric,
                last_kin,
            )
            p = self.active[pid]
            p.period = cur_period          # allow period to update if detected minimal period changes
            p.nodes = cur_nodes
            p.last_tick = tick
            p.kinematics = kin

        # Retire unmatched previous particles
        for i in prev_unmatched:
            _, _, pid = prev_clusters[i]
            self.archive[pid] = self.active.pop(pid)

        # Create new particles for unmatched current clusters
        for j in curr_unmatched:
            cur_period, cur_nodes = curr_clusters[j]
            pid = self._next_pid
            self._next_pid += 1
            kin = calculate_kinematics(
                Particle(pid, cur_period, cur_nodes, tick, tick, {}),
                self.site.atlas,
                self.site.metric,
                None,
            )
            self.active[pid] = Particle(pid, cur_period, cur_nodes, tick, tick, kin)

        return self.active
