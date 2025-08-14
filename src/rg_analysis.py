#!/usr/bin/env python3
# src/rg_analysis.py
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import networkx as nx


def _hop_ball(graph: nx.DiGraph, center: int, radius: int) -> Set[int]:
    """Nodes within <= radius hops from center, treating edges undirected."""
    if radius <= 0:
        return {int(center)}
    seen = {int(center)}
    frontier = {int(center)}
    for _ in range(radius):
        nxt = set()
        for u in frontier:
            for v in graph.predecessors(u):
                if v not in seen:
                    seen.add(int(v)); nxt.add(int(v))
            for v in graph.successors(u):
                if v not in seen:
                    seen.add(int(v)); nxt.add(int(v))
        if not nxt:
            break
        frontier = nxt
    return seen


def _degree_curvature_proxy(graph: nx.DiGraph, R_cap: Optional[int] = None) -> np.ndarray:
    """
    Lightweight curvature proxy: kappa(v) = R_cap - out_degree(v).
    If R_cap is None, use the max observed out-degree as the cap.
    """
    nodes = list(graph.nodes)
    out_deg = np.array([graph.out_degree(n) for n in nodes], dtype=float)
    cap = float(R_cap) if R_cap is not None else float(np.max(out_deg) if len(out_deg) else 0.0)
    return (cap - out_deg).astype(float)


def _curvature_per_node(site, k: int) -> np.ndarray:
    """
    Try to import a proper per-node curvature from geometry_report; otherwise
    return a degree-based proxy aligned with node order.
    """
    nodes = list(site.graph.nodes)
    try:
        from src.geometry_report import curvature_per_node  # expected signature (site, k) -> np.ndarray
        arr = curvature_per_node(site, k)
        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] == len(nodes):
            return arr
    except Exception:
        pass
    # Fallback
    return _degree_curvature_proxy(site.graph, R_cap=site.config["tags"]["max_out_degree_R"])


def _coarse_avg_over_balls(values: np.ndarray, graph: nx.DiGraph, k: int) -> np.ndarray:
    """
    For each node v, average 'values' over its hop-ball radius k.
    Returns an array aligned with graph.nodes order.
    """
    nodes = list(graph.nodes)
    out = np.zeros(len(nodes), dtype=float)
    for i, v in enumerate(nodes):
        ball = _hop_ball(graph, int(v), k)
        if not ball:
            out[i] = float(values[i])
        else:
            idxs = [nodes.index(b) for b in ball]  # small graphs ok; can optimize if needed
            out[i] = float(np.mean(values[idxs]))
    return out


def estimate_g_k(site, memory_density: np.ndarray, k: int) -> float:
    """
    Estimate g_k via block-averaged ratio kappa_k / rho_k.
    Returns NaN if insufficient data.
    """
    graph = site.graph
    nodes = list(graph.nodes)
    if len(nodes) == 0:
        return float("nan")

    # Curvature per node at scale k
    kappa = _curvature_per_node(site, k)  # (N,)
    # Memory density averaged over hop-balls radius k
    if memory_density.shape[0] != len(nodes):
        # attempt alignment by node id order (assumes 0..N-1 contiguous)
        md = np.zeros(len(nodes), dtype=float)
        for i, n in enumerate(nodes):
            if int(n) < memory_density.shape[0]:
                md[i] = float(memory_density[int(n)])
        memory_density = md.astype(float, copy=False)
    rho_k = _coarse_avg_over_balls(memory_density.astype(float), graph, k)

    eps = 1e-9
    mask = rho_k > eps
    if not np.any(mask):
        return float("nan")
    ratios = kappa[mask] / np.maximum(rho_k[mask], eps)
    return float(np.mean(ratios))


def estimate_g_series(site, memory_density: np.ndarray, k_values: Iterable[int]) -> Dict[int, float]:
    """
    Compute g_k for multiple k values; returns {k: g_k}.
    """
    out: Dict[int, float] = {}
    for k in k_values:
        try:
            out[int(k)] = float(estimate_g_k(site, memory_density, int(k)))
        except Exception:
            out[int(k)] = float("nan")
    return out
