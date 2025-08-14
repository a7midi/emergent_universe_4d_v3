"""
Depth-scaled quasi-metric d_∞ (Paper III §2.7)

For a finite acyclic causal site with layer(u) = depth(u):

    d_∞(u, v) = 2^(-depth(v))   if v is reachable from u (including u=v)
                +∞               otherwise.

Note the asymmetry: once reachability(u→v) is known, d_∞ depends only on v.
We therefore precompute:
  • depth_val[v] = 2^(-depth(v))
  • reach[u]     = { v : there is a directed path u →* v } ∪ {u}

A symmetric (true) metric is obtained via:
    r(u, v) = max{ d_∞(u, v), d_∞(v, u) }.
"""

from __future__ import annotations
from typing import Dict, Set

import networkx as nx


class DepthMetric:
    def __init__(self, causal_site: "CausalSite"):
        g: nx.DiGraph = causal_site.graph
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("DepthMetric requires a DAG (finite acyclic causal site).")

        # depth/layer lookup and the 2^(-depth) values
        self._depth: Dict[int, int] = {int(v): int(g.nodes[v]["layer"]) for v in g.nodes}
        self._val: Dict[int, float] = {v: 2.0 ** (-d) for v, d in self._depth.items()}

        # reachability sets: reachable(u) contains u and all descendants of u
        # networkx.descendants is fast enough for our graph sizes and is deterministic.
        self._reach: Dict[int, Set[int]] = {}
        for u in g.nodes:
            u = int(u)
            self._reach[u] = set(nx.descendants(g, u))
            self._reach[u].add(u)

    # ------------------------------------------------------------------ #
    def d_infty(self, u: int, v: int) -> float:
        """Return d_∞(u, v) as defined above."""
        u = int(u); v = int(v)
        return self._val[v] if v in self._reach[u] else float("inf")

    def get_symmetric_radius(self, u: int, v: int) -> float:
        """r(u, v) = max{ d_∞(u, v), d_∞(v, u) } — a true metric."""
        u = int(u); v = int(v)
        # Early exits for identity / same depth cases are cheap, but not necessary.
        r_uv = self._val[v] if v in self._reach[u] else float("inf")
        r_vu = self._val[u] if u in self._reach[v] else float("inf")
        return r_uv if r_uv >= r_vu else r_vu
