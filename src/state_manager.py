"""
state_manager.py

VERSION 44 (Deterministic & Provenanced):
- Single seeded RNG shared with CausalSite (no secondary global RNGs).
- InjectiveFusionTable is pre-seeded (surjective from tick 0) and exposes a signature.
- Predecessor lists are sorted before fusion (platform-independent determinism).
- Hidden-layer updates use the SAME RNG stream (no accidental reseeding).
- Retains drifting clocks, causal consistency, and memory-density tracking.
"""

from __future__ import annotations
from math import gcd
from typing import Dict, Tuple, Optional

import numpy as np
import hashlib

from src.causal_site import CausalSite


# --------------------------------------------------------------------------- #
# Injective fusion (history-dependent, pre-seeded for surjectivity)
# --------------------------------------------------------------------------- #
class InjectiveFusionTable:
    """'injective' fusion mode: complex, history-dependent, and surjective."""
    def __init__(self, q: int, rng: np.random.Generator):
        self.q: int = int(q)
        self.rng: np.random.Generator = rng
        self.mapping: Dict[Tuple[int, ...], int] = {}

        # Pre-seed with a permutation of all q tags to guarantee surjectivity
        self.preseed_values: np.ndarray = self.rng.permutation(self.q)
        self.preseed_idx: int = 0

        # After preseed is exhausted, wrap a simple counter over Z_q
        self.next_tag_internal: int = 0

        # Stable signature for provenance (exported in run metadata)
        h = hashlib.sha256(self.preseed_values.tobytes())
        self._signature_hex: str = h.hexdigest()

    @property
    def signature(self) -> str:
        """Hex digest (sha256) of the pre-seed permutation; stable provenance."""
        return self._signature_hex

    def fuse(self, pred_tags_tuple: Tuple[int, ...], receiver_id: int) -> int:
        """
        Deterministic mapping keyed by (ordered predecessor tags, receiver id).
        First q unseen keys pull from the preseed permutation, guaranteeing a
        surjective image onto {0,...,q-1}. Afterwards we cycle a counter mod q.
        """
        key = pred_tags_tuple + (int(receiver_id),)

        if key not in self.mapping:
            if self.preseed_idx < self.q:
                tag = int(self.preseed_values[self.preseed_idx])
                self.preseed_idx += 1
            else:
                tag = int(self.next_tag_internal)
                self.next_tag_internal = (self.next_tag_internal + 1) % self.q
            self.mapping[key] = tag

        return self.mapping[key]


# --------------------------------------------------------------------------- #
# State manager
# --------------------------------------------------------------------------- #
class StateManager:
    """
    Evolves node tags over the causal site using a deterministic tag-fusion rule.
    """

    def __init__(self, causal_site: CausalSite, config: dict):
        if not config:
            raise ValueError("Configuration could not be loaded. Aborting.")

        self.causal_site: CausalSite = causal_site
        self.config = config

        self.num_nodes: int = self.causal_site.graph.number_of_nodes()
        self.q: int = int(self.config["tags"]["alphabet_size_q"])

        # Use the SAME RNG as the causal site for full-run reproducibility
        self.rng: np.random.Generator = self.causal_site.rng

        # State vectors
        self.state: np.ndarray = np.zeros(self.num_nodes, dtype=np.int32)
        self.tick_counter: int = 0

        # Hidden layer (single layer, as per current design)
        sim_cfg = self.config.get("simulation", {})
        hide_layer_index: Optional[int] = sim_cfg.get("hide_layer_index")
        self.hidden_nodes = set()
        if hide_layer_index is not None and hide_layer_index >= 0:
            self.hidden_nodes = set(self.causal_site.nodes_by_layer.get(hide_layer_index, []))

        # Hidden-params (affine maps over Z_q), using the SAME RNG stream
        self.hidden_params: Dict[int, Tuple[int, int]] = {}
        if self.hidden_nodes:
            for n in self.hidden_nodes:
                # Pick multiplier coprime to q
                mult = int(self.rng.integers(2, max(self.q, 3)))
                while gcd(mult, self.q) != 1:
                    mult = int((mult + 1) % self.q or 1)
                add = int(self.rng.integers(0, self.q))
                self.hidden_params[n] = (mult, add)

        # Fusion rule selection
        self.fusion_mode: str = self.config["tags"].get("fusion_mode", "injective")
        self.fusion_table: Optional[InjectiveFusionTable] = None
        if self.fusion_mode == "injective":
            self.fusion_table = InjectiveFusionTable(self.q, self.rng)

        # Memory density
        det_cfg = self.config.get("detector", {})
        self.mem_window: int = int(det_cfg.get("memory_window", 1024))
        self.memory_density: np.ndarray = np.zeros(self.num_nodes, dtype=np.uint16)

        # Initialise tags
        self.initialize_state()

    # --------------------------- lifecycle --------------------------------- #
    def initialize_state(self) -> None:
        """Initialises node tags using the manager's RNG (reproducible)."""
        self.state = self.rng.integers(0, self.q, size=self.num_nodes, dtype=np.int32)

    def _fusion(self, predecessor_tags: Tuple[int, ...], node_id: int) -> int | None:
        if not predecessor_tags:
            return None
        if self.fusion_mode == "injective":
            return self.fusion_table.fuse(predecessor_tags, node_id)  # type: ignore[union-attr]
        if self.fusion_mode == "sum_mod_q":
            return int(sum(predecessor_tags) % self.q)
        if self.fusion_mode == "quadratic":
            s_mod = int(sum(predecessor_tags) % self.q)
            return int((s_mod * s_mod) % self.q)
        raise ValueError(f"Unknown fusion_mode: '{self.fusion_mode}' in config.yaml")

    def tick(self) -> None:
        """Apply one causally consistent deterministic update."""
        self.tick_counter += 1
        state_at_t = self.state
        next_state = state_at_t.copy()

        # ---- Phase 1: Hidden layer affine updates over Z_q ----
        for node_id in self.hidden_nodes:
            mult, add = self.hidden_params[node_id]
            # occasional drift to avoid trivial cycles
            if self.tick_counter > 0 and self.tick_counter % 11 == 0:
                add = (add + node_id) % self.q
                new_mult = (mult + 1 + node_id) % self.q or 1
                while gcd(new_mult, self.q) != 1:
                    new_mult = (new_mult + 1) % self.q or 1
                mult = new_mult
                self.hidden_params[node_id] = (mult, add)
            next_state[node_id] = (mult * state_at_t[node_id] + add) % self.q

        # ---- Phase 2: Observable layers, in increasing layer order ----
        if self.causal_site.nodes_by_layer:
            max_layer = max(self.causal_site.nodes_by_layer.keys())
            for layer_index in range(1, max_layer + 1):
                for node_id in self.causal_site.nodes_by_layer.get(layer_index, []):
                    if node_id in self.hidden_nodes:
                        continue
                    # Canonical ordering of predecessors for determinism
                    predecessors = tuple(sorted(self.causal_site.get_predecessors(node_id)))
                    if not predecessors:
                        continue
                    predecessor_tags = tuple(int(state_at_t[p]) for p in predecessors)
                    new_tag = self._fusion(predecessor_tags, node_id)
                    if new_tag is not None:
                        next_state[node_id] = new_tag

        # ---- Memory density update (bounded counter per node) ----
        changed = next_state != state_at_t
        if np.any(changed):
            self.memory_density[changed] = np.minimum(
                self.memory_density[changed] + 1, self.mem_window
            )
        if np.any(~changed):
            self.memory_density[~changed] = np.maximum(
                self.memory_density[~changed] - 1, 0
            )

        self.state = next_state

    # ----------------------------- getters --------------------------------- #
    def get_current_state(self) -> np.ndarray:
        return self.state

    def get_memory_density(self) -> np.ndarray:
        return self.memory_density

    def get_fusion_signature(self) -> Optional[str]:
        """Expose the injective fusion preseed signature for run provenance."""
        if self.fusion_table and hasattr(self.fusion_table, "signature"):
            return self.fusion_table.signature
        return None
