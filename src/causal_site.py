# src/causal_site.py
from __future__ import annotations
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from src.depth_metric import DepthMetric
from src.chart_atlas import ChartAtlas


class CausalSite:
    """
    Generates a finite acyclic causal site (layered DAG) together with
    emergent-geometry helpers:  self.metric  and  self.atlas.

    Changes vs. previous version:
      • All randomness comes from ONE seeded NumPy Generator (self.rng).
      • No usage of the `random` stdlib module (full reproducibility).
      • Out-degree trimming and parent backfilling use self.rng, not globals.
      • get_predecessors returns a tuple (not an iterator), avoiding multi-pass bugs.
    """

    # -------------------------------------------------------------- #
    def __init__(self, config: Dict):
        self.config = config
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes_by_layer: Dict[int, List[int]] = {}
        self.node_positions: Dict[int, np.ndarray] = {}  # legacy fall-back
        self.metric: DepthMetric | None = None
        self.atlas: ChartAtlas | None = None

        # Single RNG used everywhere this object needs randomness.
        sim_cfg = (config or {}).get("simulation", {})
        seed = sim_cfg.get("seed", 42)
        self.rng: np.random.Generator = np.random.default_rng(seed)

    # -------------------------------------------------------------- #
    #  1. generate graph                                             #
    # -------------------------------------------------------------- #
    def generate_graph(self) -> None:
        cfg = self.config["causal_site"]
        layers: int = int(cfg["layers"])
        avg: float = float(cfg["avg_nodes_per_layer"])
        edge_p: float = float(cfg["edge_probability"])
        max_back: int = int(cfg["max_lookback_layers"])
        R: int = int(self.config["tags"]["max_out_degree_R"])

        print("Generating causal site graph...")
        node_counter = 0

        # --- create layered vertices ---
        for layer_idx in range(layers):
            # Poisson draw from the SAME generator for determinism
            n_in_layer = int(self.rng.poisson(avg))
            self.nodes_by_layer[layer_idx] = []
            for _ in range(n_in_layer):
                self.graph.add_node(node_counter, layer=layer_idx)
                self.nodes_by_layer[layer_idx].append(node_counter)
                node_counter += 1

            # --- connect to previous ≤ max_back layers ---
            for nid in self.nodes_by_layer[layer_idx]:
                for back in range(1, max_back + 1):
                    prev_layer = layer_idx - back
                    if prev_layer < 0:
                        break
                    parents = self.nodes_by_layer.get(prev_layer, [])
                    if not parents:
                        continue
                    # Bernoulli trials via same RNG
                    # Vectorized coin-flips for speed and determinism
                    if parents:
                        flags = self.rng.random(len(parents)) < edge_p
                        for parent, keep in zip(parents, flags):
                            if keep:
                                self.graph.add_edge(parent, nid)

        # --- Enforce out-degree ≤ R deterministically ---
        print(f"Enforcing maximum successor count (R) of {R}...")
        for parent in list(self.graph.nodes):
            succ = list(self.graph.successors(parent))
            if len(succ) > R:
                # Choose which edges to REMOVE using the same RNG
                drop = self.rng.choice(succ, size=len(succ) - R, replace=False)
                for child in drop:
                    self.graph.remove_edge(parent, int(child))

        # --- Ensure every *visible* node has at least one parent ---
        hidden_layer = int(self.config["simulation"]["hide_layer_index"])
        print("Safeguarding against isolated visible nodes...")
        for layer_idx, ids in self.nodes_by_layer.items():
            if layer_idx <= hidden_layer:
                continue
            if not self.nodes_by_layer.get(layer_idx - 1):
                continue  # nothing to attach to
            for nid in ids:
                if self.graph.in_degree(nid) == 0:
                    parent = int(self.rng.choice(self.nodes_by_layer[layer_idx - 1]))
                    self.graph.add_edge(parent, int(nid))

        print(f"Graph generation complete. Total nodes: {self.graph.number_of_nodes()}")

    # -------------------------------------------------------------- #
    #  2. build emergent geometry                                    #
    # -------------------------------------------------------------- #
    def build_emergent_geometry(self) -> None:
        """
        Creates:
            self.metric – DepthMetric   (fast quasi-metric)
            self.atlas  – ChartAtlas    (4-D positions)
        """
        gcfg = self.config.get("geometry", {})
        chart_k = int(gcfg.get("chart_scale_k", 4))
        gh_tol = float(gcfg.get("gh_tolerance", 0.05))

        print(f"Building chart atlas with radius L_k={2 ** (-chart_k)} (k={chart_k})...")

        # metric first
        self.metric = DepthMetric(self)
        # then atlas (needs metric)
        self.atlas = ChartAtlas(self, self.metric, chart_k, gh_tol)

        # store legacy 3-D positions for fall-back plotting
        for nid in self.graph.nodes:
            self.node_positions[int(nid)] = self.atlas.position(nid)[:3]

        print("Chart atlas construction complete.")

    # -------------------------------------------------------------- #
    #  convenience helper used by ParticleDetector & GeometryStats   #
    # -------------------------------------------------------------- #
    def get_predecessors(self, nid: int) -> Tuple[int, ...]:
        # Return a tuple to avoid iterator re-use issues upstream
        return tuple(self.graph.predecessors(nid))
