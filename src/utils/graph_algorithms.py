# utils/graph_algorithms.py
# --- UPGRADED TO CLUSTER NODES, NOT CELLS ---

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set

class _UnionFind:
    """A standard Union-Find implementation."""
    __slots__ = ("p", "rank")

    def __init__(self, elements):
        self.p: Dict[int, int] = {e: e for e in elements}
        self.rank: Dict[int, int] = {e: 0 for e in elements}

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return
        if self.rank[xr] < self.rank[yr]:
            self.p[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.p[yr] = xr
        else:
            self.p[yr] = xr
            self.rank[xr] += 1

def find_connected_clusters(looping_nodes: Set[int], causal_site) -> List[List[int]]:
    """
    Finds connected components among a set of looping nodes within the causal graph.
    
    Args:
        looping_nodes (Set[int]): Set of node IDs that are in a periodic loop.
        causal_site (CausalSite): The main CausalSite object.

    Returns:
        List[List[int]]: A list of clusters, each a list of connected node IDs.
    """
    if not looping_nodes:
        return []

    # --- UPGRADE: Union-Find now operates directly on looping node IDs.
    uf = _UnionFind(looping_nodes)

    # --- UPGRADE: Iterate over all edges to find connections between looping nodes.
    for u, v in causal_site.graph.edges():
        # Check if both ends of an edge are in the looping set.
        if u in looping_nodes and v in looping_nodes:
            uf.union(u, v)

    # --- UPGRADE: Collect clusters of node IDs.
    clusters_dict: Dict[int, List[int]] = defaultdict(list)
    for node_id in looping_nodes:
        clusters_dict[uf.find(node_id)].append(node_id)

    return list(clusters_dict.values())