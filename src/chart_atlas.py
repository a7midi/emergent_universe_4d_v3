"""
ChartAtlas – hop-based neighbourhood implementation
Local charts are defined by hop distance (undirected BFS) rather than by the
symmetric radius metric (which is ∞ for most pairs in a DAG).

Pipeline (per centre):
  1) Build hop-ball of radius H = H(k0).
  2) Compute pairwise symmetric radii within the ball; replace non-finite with a
     conservative large value derived from the finite entries.
  3) Embed into R^3 via MDS(dissimilarity="precomputed").
  4) If ≥3 points already placed in the global coords, attempt Procrustes align
     to those; accept only if RMS error ≤ gh_tol; else just translate to the
     existing centroid (no rotation).
  5) Write (x,y,z, τ) with τ := −layer.

Exposes:
  • id_map:        node id -> row index
  • global_coords: (N, 4) array of [x,y,z,τ]
  • position(nid): returns the 4-vector for nid
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Set

import itertools
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

from src.depth_metric import DepthMetric


class ChartAtlas:
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        causal_site: "CausalSite",
        dmetric: DepthMetric,
        chart_scale_k: int = 4,
        gh_tol: float = 0.05,
    ):
        self.site = causal_site
        self.metric = dmetric
        self.k0 = int(chart_scale_k)
        self.gh_tol = float(gh_tol)

        # Deterministic node order
        self.node_ids: List[int] = [int(n) for n in self.site.graph.nodes]
        self.id_map: Dict[int, int] = {nid: i for i, nid in enumerate(self.node_ids)}
        self.global_coords = np.full((len(self.node_ids), 4), np.nan, dtype=np.float64)

        self._build_atlas()

    # ------------------------------------------------------------------ #
    def _max_hops(self) -> int:
        """Heuristic: shrink neighbourhoods as k0 grows (k0>=4 → 2 hops)."""
        return max(2, 6 - self.k0)

    # ------------------------------------------------------------------ #
    def _ball_by_hops(self, centre: int) -> List[int]:
        """BFS up to max_hops with edges treated as undirected."""
        max_h = self._max_hops()
        centre = int(centre)
        visited: Set[int] = {centre}
        ball = [centre]
        q = deque([(centre, 0)])
        succ = self.site.graph.successors
        pred = self.site.graph.predecessors
        while q:
            nid, h = q.popleft()
            if h >= max_h:
                continue
            for nb in itertools.chain(succ(nid), pred(nid)):
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    ball.append(nb)
                    q.append((nb, h + 1))
        return ball

    # ------------------------------------------------------------------ #
    def _pairwise_radius_matrix(self, nodes: List[int]) -> np.ndarray:
        """Symmetric (len(nodes) x len(nodes)) matrix of r(u,v) with robust fill."""
        n = len(nodes)
        D = np.zeros((n, n), dtype=float)
        finite_vals = []
        for i, u in enumerate(nodes):
            for j in range(i + 1, n):
                d = self.metric.get_symmetric_radius(u, nodes[j])
                if np.isfinite(d):
                    D[i, j] = D[j, i] = float(d)
                    finite_vals.append(float(d))
                else:
                    D[i, j] = D[j, i] = np.nan

        if finite_vals:
            # Fill unknown distances with a conservative large value
            fill = 1.5 * float(np.max(finite_vals))
        else:
            fill = 1.0  # small chart with no finite info—degenerate but consistent
        D = np.where(np.isfinite(D), D, fill)
        np.fill_diagonal(D, 0.0)
        return D

    # ------------------------------------------------------------------ #
    def _embed_and_place(self, centre: int) -> Set[int]:
        nodes = self._ball_by_hops(centre)
        if len(nodes) < 4:
            return set()

        idxs = [self.id_map[n] for n in nodes]

        # Pairwise symmetric radius inside the small ball
        D = self._pairwise_radius_matrix(nodes)

        # MDS embedding to R^3 (deterministic with fixed random_state)
        X3 = MDS(
            n_components=3,
            dissimilarity="precomputed",
            normalized_stress=False,
            random_state=0,
            max_iter=200,
            n_init=4,
        ).fit_transform(D)

        # Align with already-stitched coords if ≥3 in common
        existing = [i for i in idxs if not np.isnan(self.global_coords[i, 0])]
        if len(existing) >= 3:
            # Correspondences by index within this local chart
            A = X3[[idxs.index(i) for i in existing]]
            B = self.global_coords[existing, :3]
            try:
                R, _ = orthogonal_procrustes(A, B)
                A_aligned = A @ R
                # RMS error (Procrustes)
                err = float(np.sqrt(np.mean((A_aligned - B) ** 2)))
                if err <= self.gh_tol:
                    X3 = X3 @ R + (B.mean(0) - A_aligned.mean(0))
                else:
                    # If the fit is poor, keep X3's internal geometry but simply
                    # translate to the existing centroid (no rotation).
                    X3 = X3 + (B.mean(0) - A.mean(0))
            except Exception:
                # Robust fallback: translation only
                X3 = X3 + (B.mean(0) - A.mean(0))

        placed = set()
        for loc, row in zip(X3, idxs):
            if np.isnan(self.global_coords[row, 0]):
                τ = -int(self.site.graph.nodes[self.node_ids[row]]["layer"])
                self.global_coords[row] = np.array([loc[0], loc[1], loc[2], τ], dtype=float)
                placed.add(row)
        return placed

    # ------------------------------------------------------------------ #
    def _build_atlas(self):
        undirected = self.site.graph.to_undirected()
        comps = [set(map(int, comp)) for comp in nx.connected_components(undirected)]

        pbar = tqdm(total=len(self.node_ids), desc="Stitching Charts")
        stitched: Set[int] = set()

        for comp in comps:
            todo = deque([next(iter(comp))])
            while todo:
                centre = int(todo.popleft())
                if centre in stitched:
                    continue
                newly = self._embed_and_place(centre)
                if newly:
                    stitched |= newly
                    pbar.update(len(newly))
                    # Queue neighbours of all *newly placed nodes* (by node id)
                    for row in newly:
                        nid = self.node_ids[row]
                        for nb in itertools.chain(
                            self.site.graph.successors(nid),
                            self.site.graph.predecessors(nid),
                        ):
                            nb = int(nb)
                            if nb in comp and (nb not in stitched):
                                todo.append(nb)
        pbar.close()

        # Any remaining unplaced nodes → fallback (0,0,0,τ)
        for nid in self.node_ids:
            row = self.id_map[nid]
            if np.isnan(self.global_coords[row, 0]):
                τ = -int(self.site.graph.nodes[nid]["layer"])
                self.global_coords[row] = np.array([0.0, 0.0, 0.0, τ], dtype=float)

    # ------------------------------------------------------------------ #
    def position(self, nid: int) -> np.ndarray:
        """Return 4-vector [x,y,z,τ] for node nid."""
        return self.global_coords[self.id_map[int(nid)]]
