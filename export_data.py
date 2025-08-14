#!/usr/bin/env python3
from __future__ import annotations
import gzip
import json
import math
import pathlib
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
from tqdm import tqdm

from src.config import CONFIG
from src.causal_site import CausalSite
from src.state_manager import StateManager
from src.particle_detector import ParticleDetector

# ----------------------------------------------------------------------------- #
# Output paths
# ----------------------------------------------------------------------------- #
OUT = pathlib.Path("results")
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def to_py(obj: Any) -> Any:
    """Deep-convert NumPy scalars and drop non-finite floats."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    return obj


def serialisable_particle(p) -> dict:
    d = to_py(p.__dict__)
    d["nodes"] = [int(n) for n in p.nodes]
    d["num_nodes"] = len(d["nodes"])
    return d


def _period_histogram(live_particles: Dict[int, Any]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for p in live_particles.values():
        k = str(int(p.period))
        hist[k] = hist.get(k, 0) + 1
    return hist


def _curvature_proxy_or_none(site: CausalSite) -> Optional[float]:
    """
    Hook: try to compute a very lightweight curvature proxy from the atlas/metric.
    If a richer function exists in geometry_report.py, use it; otherwise return None.
    """
    try:
        from src.geometry_report import curvature_proxy  # optional
        val = curvature_proxy(site)  # expected to return a float
        if isinstance(val, (int, float)) and math.isfinite(val):
            return float(val)
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------- #
# Static substrate
# ----------------------------------------------------------------------------- #
def dump_static(site: CausalSite) -> None:
    data = {
        "nodes": {
            str(n): {
                "layer": int(meta["layer"]),
                "position": list(map(float, site.atlas.position(n))),
            }
            for n, meta in site.graph.nodes(data=True)
        },
        "edges": [[int(u), int(v)] for u, v in site.graph.edges()],
    }
    (OUT / "static_universe.json").write_text(json.dumps(data, indent=2))
    print("✓ static_universe.json written")


# ----------------------------------------------------------------------------- #
# Dynamic log with summaries + provenance
# ----------------------------------------------------------------------------- #
def dump_log(site: CausalSite, sm: StateManager, det: ParticleDetector) -> None:
    sim = CONFIG["simulation"]
    total = int(sim["total_ticks"])
    interval = max(1, int(sim.get("log_interval", 1)))
    verbose = bool(sim.get("verbose", False))

    out_cfg = CONFIG.get("output", {})
    use_gzip = bool(out_cfg.get("gzip", False))
    schema = str(out_cfg.get("schema", "v1"))
    log_path = OUT / ("simulation_log.jsonl.gz" if use_gzip else "simulation_log.jsonl")

    # Experiments toggles
    exp_cfg = CONFIG.get("experiments", {})
    chsh_cfg = (exp_cfg or {}).get("chsh", {}) or {}
    rg_cfg = (exp_cfg or {}).get("rg", {}) or {}
    chsh_enabled = bool(chsh_cfg.get("enabled", False))
    rg_enabled = bool(rg_cfg.get("enabled", False))

    # Optional experiment initialisation
    chsh = None
    if chsh_enabled:
        try:
            from src.chsh_experiment import CHSHExperiment, CHSHSettings
            chsh = CHSHExperiment(
                site,
                CHSHSettings(
                    pairs_per_measurement=int(chsh_cfg.get("pairs_per_measurement", 64)),
                    ball_radius=int(chsh_cfg.get("ball_radius", 2)),
                    a_offset=int(chsh_cfg.get("a_offset", 0)),
                    a_prime_offset=int(chsh_cfg.get("a_prime_offset", 1)),
                    b_offset=int(chsh_cfg.get("b_offset", 0)),
                    b_prime_offset=int(chsh_cfg.get("b_prime_offset", 1)),
                ),
            )
        except Exception as e:
            print(f"[warn] CHSH experiment unavailable ({e}); continuing without S.")

    rg_k_values = None
    if rg_enabled:
        # we import lazily inside the loop as well for safety; but cache flags here
        ks = rg_cfg.get("k_values", None)
        if isinstance(ks, (list, tuple)):
            try:
                rg_k_values = [int(x) for x in ks]
            except Exception:
                rg_k_values = None

    # Provenance header (separate file, stable schema)
    meta = {
        "schema": schema,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": CONFIG.get("simulation", {}).get("seed", 42),
        "config_snapshot": CONFIG,  # already a plain dict
        "fusion_signature": sm.get_fusion_signature(),
        "total_ticks": total,
        "notes": "JSONL stream contains per-tick frames with optional 'summary' key.",
    }
    (OUT / "run_meta.json").write_text(json.dumps(to_py(meta), indent=2))
    print("✓ run_meta.json written")

    # Optional curvature proxy (constant per run or cheap to compute each tick)
    curv_const = _curvature_proxy_or_none(site)

    opener = gzip.open if use_gzip else open
    with opener(log_path, "wt", encoding="utf-8") as fp, tqdm(range(total), desc="ticks") as bar:
        particles_in_window = 0
        longest_lifetime_so_far = 0

        for t in bar:
            # Step the world and detect particles
            sm.tick()
            live = det.detect(sm.get_current_state(), t)

            # Track longest lifetime seen so far (active ∪ archive)
            for p in live.values():
                if p.lifetime > longest_lifetime_so_far:
                    longest_lifetime_so_far = p.lifetime
            for p in det.archive.values():
                if p.lifetime > longest_lifetime_so_far:
                    longest_lifetime_so_far = p.lifetime

            # Memory-density stats (expose paper-facing signal)
            md = sm.get_memory_density()
            md_mean = float(np.mean(md)) if md.size else 0.0
            md_var = float(np.var(md)) if md.size else 0.0

            # Experiments (sampled at the logging cadence to keep things light)
            chsh_S = None
            chsh_detail = None
            if chsh is not None and ((t + 1) % interval == 0 or interval == 1):
                try:
                    res = chsh.measure(sm.get_current_state(), q=CONFIG["tags"]["alphabet_size_q"])
                    chsh_S = float(res.get("S", float("nan")))
                    chsh_detail = {k: float(res[k]) for k in ("Eab", "Eabp", "Eapb", "Eapbp") if k in res}
                except Exception as e:
                    chsh_S = None
                    chsh_detail = None

            g_k = None
            g_series = None
            if rg_enabled and ((t + 1) % interval == 0 or interval == 1):
                try:
                    from src import rg_analysis
                    # main k from geometry
                    k_main = int(CONFIG.get("geometry", {}).get("chart_scale_k", 4))
                    g_k = float(rg_analysis.estimate_g_k(site, md, k=k_main))
                    if rg_k_values:
                        g_series = {int(k): float(v) for k, v in rg_analysis.estimate_g_series(site, md, rg_k_values).items()}
                except Exception as e:
                    g_k = None
                    g_series = None

            # Per-tick summary payload (won't break existing dashboards)
            summary = {
                "live_particles": int(len(live)),
                "period_histogram": _period_histogram(live),
                "longest_lifetime_so_far": int(longest_lifetime_so_far),
                "memory_density_mean": md_mean,
                "memory_density_var": md_var,
                "curvature_proxy": curv_const,
                "chsh_S": chsh_S,
                "chsh_detail": chsh_detail,
                "g_k": g_k,
                "g_series": g_series,
            }

            # Emit the frame
            frame = {
                "tick": int(t),
                "particles": [serialisable_particle(p) for p in live.values()],
                "summary": to_py(summary),
            }
            fp.write(json.dumps(frame) + "\n")

            # Console status (optional)
            particles_in_window += len(live)
            if verbose and (t + 1) % interval == 0:
                tqdm.write(
                    f"[t={t:>6}] detected in last {interval:>3} ticks: {particles_in_window}"
                )
                particles_in_window = 0  # reset for next window

    print(f"✓ {log_path.name} written  (contains full 0…{total - 1} range)")


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The default value of `n_init`",
        category=FutureWarning,
    )

    site = CausalSite(CONFIG)
    site.generate_graph()
    site.build_emergent_geometry()
    dump_static(site)

    sm = StateManager(site, CONFIG)
    det = ParticleDetector(site, sm, CONFIG)
    dump_log(site, sm, det)


if __name__ == "__main__":
    main()
