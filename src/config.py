"""
config.py

Loads config.yaml once, deep-merges with sane defaults, and exposes CONFIG.
Also provides a get_config() helper if you prefer lazy access.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict

import yaml


_DEFAULTS: Dict[str, Any] = {
    "simulation": {
        "total_ticks": 2000,
        "seed": 42,
        "hide_layer_index": 0,
        "log_interval": 100,
        "verbose": False,
    },
    "output": {
        # when true, writes results/simulation_log.jsonl.gz instead of .jsonl
        "gzip": False,
        # schema version for run_meta.json (bump when you change on-disk format)
        "schema": "v1",
    },
    "causal_site": {
        "layers": 100,
        "avg_nodes_per_layer": 32,
        "edge_probability": 0.15,
        "max_lookback_layers": 4,
    },
    "tags": {
        "alphabet_size_q": 17,
        "max_out_degree_R": 5,
        "fusion_mode": "injective",  # 'injective' | 'quadratic' | 'sum_mod_q'
    },
    "detector": {
        "grid_size": 32,
        "max_history_length": 4000,
        "min_loop_period": 2,
        "min_particle_size": 3,
        "memory_window": 1024,
    },
    "geometry": {
        # k in L_k = 2^-k
        "chart_scale_k": 4,
        "gh_tolerance": 0.05,
    },
    "experiments": {
        "chsh": {
            "enabled": False,
            "pairs_per_measurement": 64,
            "ball_radius": 2,
            "a_offset": 0,
            "a_prime_offset": 1,
            "b_offset": 0,
            "b_prime_offset": 1,
        },
        "rg": {
            "enabled": False,
            # additional k values to sample besides geometry.chart_scale_k
            "k_values": [1, 2, 3, 4, 5],
        },
    },
    "visualization": {
        "enabled": False,
        "update_interval": 10,
        "save_frames": False,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[config] '{config_path}' not found; using defaults.")
        raw = {}
    except Exception as e:
        print(f"[config] Error reading '{config_path}': {e}")
        raw = {}

    cfg = _deep_merge(_DEFAULTS, raw)

    # Basic type coerce/sanity
    sim = cfg.get("simulation", {})
    sim["total_ticks"] = int(sim.get("total_ticks", 0))
    sim["seed"] = int(sim.get("seed", 42))
    sim["hide_layer_index"] = int(sim.get("hide_layer_index", 0))
    sim["log_interval"] = max(1, int(sim.get("log_interval", 1)))
    sim["verbose"] = bool(sim.get("verbose", False))

    tags = cfg.get("tags", {})
    tags["alphabet_size_q"] = int(tags.get("alphabet_size_q", 2))
    tags["max_out_degree_R"] = int(tags.get("max_out_degree_R", 2))
    tags["fusion_mode"] = str(tags.get("fusion_mode", "injective"))

    out = cfg.get("output", {})
    out["gzip"] = bool(out.get("gzip", False))
    out["schema"] = str(out.get("schema", "v1"))

    # Mirror the single simulation seed to all submodules that rely on RNGs.
    # (We read the seed in CausalSite and re-use the same Generator everywhere.)
    cfg["simulation"] = sim
    cfg["tags"] = tags
    cfg["output"] = out
    return cfg


# Global singleton used across the suite
CONFIG: Dict[str, Any] = load_config()


def get_config() -> Dict[str, Any]:
    """Return the already-loaded CONFIG (handy in REPL/tests)."""
    return CONFIG
