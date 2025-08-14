# src/chsh_experiment.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import networkx as nx


def _hop_ball(graph: nx.DiGraph, center: int, radius: int) -> Set[int]:
    """Nodes within <= radius hops from center, undirected for locality."""
    if radius <= 0:
        return {int(center)}
    seen = {int(center)}
    frontier = {int(center)}
    for _ in range(radius):
        nxt = set()
        for u in frontier:
            # treat edges undirected for "spatial" neighborhood
            for v in graph.predecessors(u):
                if v not in seen:
                    seen.add(int(v))
                    nxt.add(int(v))
            for v in graph.successors(u):
                if v not in seen:
                    seen.add(int(v))
                    nxt.add(int(v))
        if not nxt:
            break
        frontier = nxt
    return seen


def _spacelike(graph: nx.DiGraph, a: int, b: int) -> bool:
    """Return True if a and b are spacelike (no path either direction)."""
    # Fast local BFS checks; avoids materializing full transitive closure.
    try:
        if nx.has_path(graph, a, b):
            return False
    except nx.NetworkXNoPath:
        pass
    try:
        if nx.has_path(graph, b, a):
            return False
    except nx.NetworkXNoPath:
        pass
    return True


@dataclass
class CHSHSettings:
    """Settings controlling the CHSH sampling and observables."""
    # neighborhood radius around each seed node
    ball_radius: int = 2
    # how many spacelike pairs to sample per measurement
    pairs_per_measurement: int = 64
    # observable mode: currently 'parity' only (deterministic ±1)
    observable: str = "parity"
    # phase offsets for the four settings (integers mod q)
    a_offset: int = 0
    a_prime_offset: int = 1
    b_offset: int = 0
    b_prime_offset: int = 1


class CHSHExperiment:
    """
    Deterministic CHSH experiment over a layered DAG with tag states mod q.

    For each spacelike pair (A,B), define local ±1 observables by thresholding the
    offset parity of tags inside hop-balls around A and B, then average products.
    """

    def __init__(self, site, settings: Optional[CHSHSettings] = None):
        self.site = site
        self.graph: nx.DiGraph = site.graph
        self.rng: np.random.Generator = site.rng
        self.settings = settings or CHSHSettings()
        # Cache node list for sampling
        self._nodes: np.ndarray = np.fromiter((int(n) for n in self.graph.nodes), dtype=np.int64)

    # -------------------------- observables ------------------------------- #
    def _observable_ball(self, state: np.ndarray, ball: Iterable[int], offset: int, q: int) -> int:
        """
        Deterministic ±1 observable for a region: average ((tag+offset) mod q) parity,
        then return +1 if mean parity >= 0.5 else -1. (Ties break to +1.)
        """
        if self.settings.observable != "parity":
            raise ValueError("Only 'parity' observable is implemented.")
        vals = []
        for n in ball:
            tag = int(state[int(n)])
            vals.append(((tag + offset) % q) & 1)  # parity bit
        if not vals:
            return +1
        m = float(np.mean(vals))
        return +1 if m >= 0.5 else -1

    def _pick_spacelike_pair(self) -> Optional[Tuple[int, int]]:
        """Sample random node pairs until spacelike or give up after some tries."""
        N = len(self._nodes)
        if N < 2:
            return None
        for _ in range(64):  # bounded attempts
            a, b = self.rng.choice(self._nodes, size=2, replace=False)
            a, b = int(a), int(b)
            if _spacelike(self.graph, a, b):
                return a, b
        return None

    # --------------------------- measure S -------------------------------- #
    def measure(self, state: np.ndarray, q: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate CHSH S-value by averaging over randomly sampled spacelike pairs.
        Returns {'S': value, 'Eab':..., 'Eabp':..., 'Eapb':..., 'Eapbp':...}.
        """
        q = int(q if q is not None else getattr(self.site, "config", {}).get("tags", {}).get("alphabet_size_q", 2))
        s = self.settings

        sums = {"ab": 0.0, "abp": 0.0, "apb": 0.0, "apbp": 0.0}
        count = 0

        for _ in range(max(1, int(s.pairs_per_measurement))):
            pair = self._pick_spacelike_pair()
            if pair is None:
                break
            a_seed, b_seed = pair

            A = _hop_ball(self.graph, a_seed, s.ball_radius)
            B = _hop_ball(self.graph, b_seed, s.ball_radius)

            # Compute four products for the four settings
            A_a   = self._observable_ball(state, A, s.a_offset, q)
            A_ap  = self._observable_ball(state, A, s.a_prime_offset, q)
            B_b   = self._observable_ball(state, B, s.b_offset, q)
            B_bp  = self._observable_ball(state, B, s.b_prime_offset, q)

            sums["ab"]   += A_a  * B_b
            sums["abp"]  += A_a  * B_bp
            sums["apb"]  += A_ap * B_b
            sums["apbp"] += A_ap * B_bp
            count += 1

        if count == 0:
            # no valid spacelike pairs found this tick
            return {"S": float("nan"), "Eab": float("nan"), "Eabp": float("nan"), "Eapb": float("nan"), "Eapbp": float("nan")}

        Eab   = sums["ab"]   / count
        Eabp  = sums["abp"]  / count
        Eapb  = sums["apb"]  / count
        Eapbp = sums["apbp"] / count
        S = Eab + Eabp + Eapb - Eapbp
        return {"S": float(S), "Eab": float(Eab), "Eabp": float(Eabp), "Eapb": float(Eapb), "Eapbp": float(Eapbp)}
