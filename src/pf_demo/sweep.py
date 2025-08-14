# src/pf_demo/sweep.py
from __future__ import annotations

import csv
import gzip
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yaml
import networkx as nx

from .metrics import (
    ks_grid,
    summarize_invariance,
    normalize_by_median,
    hill_tail_index,
    spearman_rank_corr,
)


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class RunConfig:
    q: int
    R: int
    seed: int
    ticks: int
    layers: int
    avg_nodes_per_layer: Optional[int] = None
    edge_probability: Optional[float] = None
    max_lookback_layers: Optional[int] = None
    chart_scale_k: Optional[int] = None


@dataclass
class RunOutputs:
    run_id: str
    out_dir: Path                      # per-run working dir (contains results/)
    periods: np.ndarray
    lifetimes: np.ndarray
    sizes: np.ndarray
    atlas_speeds: np.ndarray
    graph_speeds: np.ndarray
    ks_meta: Dict[str, float]          # optional per-run quick stats


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _read_jsonl_maybe_gz(path: Path) -> Iterable[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    else:
        with path.open("rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def _vec3_norm(v) -> float:
    try:
        return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
    except Exception:
        return 0.0


def _safe_list(obj, default=None):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [] if default is None else default


# --------------------------------------------------------------------------
# Per-run helpers
# --------------------------------------------------------------------------

def _make_run_config_yaml(base_cfg_path: Path, target_dir: Path, rc: RunConfig) -> Path:
    """
    Read the project's base config.yaml, override fields for this run,
    and write a per-run config.yaml into target_dir.
    """
    with base_cfg_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    # Ensure dicts exist
    base.setdefault("simulation", {})
    base.setdefault("output", {})
    base.setdefault("causal_site", {})
    base.setdefault("tags", {})
    base.setdefault("geometry", {})
    base.setdefault("detector", {})

    # Overrides
    base["simulation"]["total_ticks"] = int(rc.ticks)
    base["simulation"]["seed"] = int(rc.seed)
    base["causal_site"]["layers"] = int(rc.layers)

    if rc.avg_nodes_per_layer is not None:
        base["causal_site"]["avg_nodes_per_layer"] = int(rc.avg_nodes_per_layer)
    if rc.edge_probability is not None:
        base["causal_site"]["edge_probability"] = float(rc.edge_probability)
    if rc.max_lookback_layers is not None:
        base["causal_site"]["max_lookback_layers"] = int(rc.max_lookback_layers)

    base["tags"]["alphabet_size_q"] = int(rc.q)
    base["tags"]["max_out_degree_R"] = int(rc.R)
    # keep your default fusion mode unless user changed:
    base["tags"]["fusion_mode"] = base["tags"].get("fusion_mode", "injective")

    # output optimization (gz is fine; you can switch off in base config if desired)
    base.setdefault("output", {})["gzip"] = base.get("output", {}).get("gzip", True)

    if rc.chart_scale_k is not None:
        base.setdefault("geometry", {})["chart_scale_k"] = int(rc.chart_scale_k)

    # Write to per-run config.yaml
    cfg_path = target_dir / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    return cfg_path


def _launch_export(project_root: Path, run_dir: Path, python_exe: Optional[str] = None) -> None:
    """
    Launch export_data.py in a subprocess with:
      - cwd = run_dir  (so it reads THIS run's config.yaml)
      - PYTHONPATH includes project_root so `src.*` imports work
    """
    python_exe = python_exe or sys.executable
    env = os.environ.copy()

    # For local parallelism, strongly limit BLAS thread fanout per process.
    # (If you already set system-wide env vars, this is harmless.)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Ensure import path finds the project's src/
    env["PYTHONPATH"] = str(project_root)
    cmd = [python_exe, str(project_root / "export_data.py")]
    proc = subprocess.run(cmd, cwd=str(run_dir), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"export_data.py failed in {run_dir} (exit {proc.returncode})")


def _load_static_graph(static_json: Path) -> nx.Graph:
    """
    Build an undirected graph from results/static_universe.json.
    """
    with static_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    edges = [(int(u), int(v)) for (u, v) in data.get("edges", [])]
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def _parse_run_outputs(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read results from one run directory.
    Returns: (periods, lifetimes, sizes, atlas_speeds, graph_speeds)
    """
    results_dir = run_dir / "results"
    static_json = results_dir / "static_universe.json"
    log_gz = results_dir / "simulation_log.jsonl.gz"
    log_plain = results_dir / "simulation_log.jsonl"
    log_path = log_gz if log_gz.exists() else log_plain
    if not static_json.exists() or not log_path.exists():
        raise FileNotFoundError(f"Missing outputs in {results_dir}")

    # Static graph (for graph-speed)
    G = _load_static_graph(static_json)

    periods, lifetimes, sizes = [], [], []
    atlas_speeds, graph_speeds = [], []

    # previous-frame cluster membership per particle id
    prev_clusters: Dict[int, set] = {}

    for frame in _read_jsonl_maybe_gz(log_path):
        ps = frame.get("particles") or []
        current_clusters: Dict[int, set] = {}
        for p in ps:
            pid = int(p.get("id", -1))
            if pid < 0:
                continue
            period = int(p.get("period", 0) or 0)
            lifetime = int(p.get("lifetime", 0) or 0)
            nodes = set(int(n) for n in _safe_list(p.get("nodes")))
            kin = p.get("kinematics") or {}
            vel = _safe_list(kin.get("velocity"), [0.0, 0.0, 0.0])
            speed = _vec3_norm(vel)

            if period > 0:
                periods.append(period)
            if lifetime > 0:
                lifetimes.append(lifetime)
            if nodes:
                sizes.append(len(nodes))
            if speed > 0:
                atlas_speeds.append(speed)

            current_clusters[pid] = nodes

        # Graph speed per particle present in both frames
        for pid, nodes_now in current_clusters.items():
            nodes_prev = prev_clusters.get(pid)
            if not nodes_prev:
                continue
            dists = []
            for u in nodes_prev:
                try:
                    # nearest-node hop distance (median)
                    min_d = None
                    for v in nodes_now:
                        d = nx.shortest_path_length(G, source=u, target=v)
                        if min_d is None or d < min_d:
                            min_d = d
                            if d == 0:
                                break
                    if min_d is not None:
                        dists.append(min_d)
                except Exception:
                    continue
            if dists:
                graph_speeds.append(float(np.median(dists)))

        prev_clusters = current_clusters

    return (
        np.asarray(periods, dtype=float),
        np.asarray(lifetimes, dtype=float),
        np.asarray(sizes, dtype=float),
        np.asarray(atlas_speeds, dtype=float),
        np.asarray(graph_speeds, dtype=float),
    )


# --------------------------------------------------------------------------
# Parallel runner: one run spec → outputs and CSV row
# --------------------------------------------------------------------------

def _run_one(spec: dict) -> Tuple[RunOutputs, Dict[str, object]]:
    """
    Execute a single run in its own folder, parse outputs, and return both
    RunOutputs and the CSV row dict.
    """
    project_root: Path = spec["project_root"]
    base_cfg: Path = spec["base_cfg"]
    sweep_dir: Path = spec["sweep_dir"]
    python_exe: Optional[str] = spec.get("python_exe")

    q = int(spec["q"]); R = int(spec["R"]); seed = int(spec["seed"])
    ticks = int(spec["ticks"]); layers = int(spec["layers"])
    rc = RunConfig(
        q=q, R=R, seed=seed, ticks=ticks, layers=layers,
        avg_nodes_per_layer=spec.get("avg_nodes_per_layer"),
        edge_probability=spec.get("edge_probability"),
        max_lookback_layers=spec.get("max_lookback_layers"),
        chart_scale_k=spec.get("chart_scale_k"),
    )
    run_id = f"q{q}_R{R}_seed{seed}"
    run_dir = sweep_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write config and launch
    _make_run_config_yaml(base_cfg, run_dir, rc)
    print(f"Running {run_id} ...")
    _launch_export(project_root, run_dir, python_exe=python_exe)

    # Parse outputs
    periods, lifetimes, sizes, atlas_spd, graph_spd = _parse_run_outputs(run_dir)

    # Quick per-run summaries (KS vs self = 0)
    ksP = 0.0
    ksS = 0.0
    alpha, k_used = hill_tail_index(lifetimes)
    rho = spearman_rank_corr(atlas_spd, graph_spd) if (atlas_spd.size and graph_spd.size) else 0.0

    outputs = RunOutputs(
        run_id=run_id,
        out_dir=run_dir,
        periods=periods,
        lifetimes=lifetimes,
        sizes=sizes,
        atlas_speeds=atlas_spd,
        graph_speeds=graph_spd,
        ks_meta={"ksP_self": float(ksP), "ksS_self": float(ksS), "hill_alpha": float(alpha or np.nan)},
    )
    row = {
        "run_id": run_id,
        "q": q,
        "R": R,
        "seed": seed,
        "ticks": ticks,
        "layers": layers,
        "n_periods": int(periods.size),
        "n_lifetimes": int(lifetimes.size),
        "n_sizes": int(sizes.size),
        "n_atlas_speeds": int(atlas_spd.size),
        "n_graph_speeds": int(graph_spd.size),
        "hill_alpha": float(alpha if np.isfinite(alpha) else 0.0),
        "hill_k": int(k_used),
        "rho_atlas_graph": float(rho),
    }
    print(f"Finished {run_id}")
    return outputs, row


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def run_grid(
    q_list: Sequence[int],
    R_list: Sequence[int],
    seeds: Sequence[int],
    *,
    ticks: int = 12000,
    layers: int = 140,
    avg_nodes_per_layer: Optional[int] = None,
    edge_probability: Optional[float] = None,
    max_lookback_layers: Optional[int] = None,
    chart_scale_k: Optional[int] = 4,
    project_root: str | Path = ".",
    out_root: Optional[str | Path] = None,
    python_exe: Optional[str] = None,
    max_workers: int = 1,   # <-- parallel jobs
) -> Tuple[List[RunOutputs], Path]:
    """
    Orchestrate a sweep of runs. Each run is executed in its own folder
    with a per-run config.yaml, then parsed and aggregated.

    Returns (runs, sweep_dir) where:
      - runs: list of RunOutputs (raw arrays for further analysis)
      - sweep_dir: results/pf_runs/<timestamp> directory containing pf_runs.csv
    """
    project_root = Path(project_root).resolve()
    base_cfg = project_root / "config.yaml"
    if not base_cfg.exists():
        raise FileNotFoundError(f"Base config.yaml not found at {base_cfg}")

    # Where to store sweep outputs (CSV + keep each run dir)
    results_root = project_root / "results" / "pf_runs"
    results_root.mkdir(parents=True, exist_ok=True)
    sweep_dir = results_root / _timestamp() if out_root is None else Path(out_root)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of specifications
    specs: List[dict] = []
    for q in q_list:
        for R in R_list:
            for seed in seeds:
                specs.append(dict(
                    q=int(q), R=int(R), seed=int(seed),
                    ticks=int(ticks), layers=int(layers),
                    avg_nodes_per_layer=avg_nodes_per_layer,
                    edge_probability=edge_probability,
                    max_lookback_layers=max_lookback_layers,
                    chart_scale_k=chart_scale_k,
                    project_root=project_root,
                    base_cfg=base_cfg,
                    sweep_dir=sweep_dir,
                    python_exe=python_exe,
                ))

    # Run in parallel (each spec spawns its own subprocess)
    runs: List[RunOutputs] = []
    csv_rows: List[Dict[str, object]] = []
    if specs:
        print(f"Launching {len(specs)} runs with max_workers={max_workers} ...")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_run_one, spec) for spec in specs]
            for i, fu in enumerate(as_completed(futs), 1):
                try:
                    out, row = fu.result()
                    runs.append(out)
                    csv_rows.append(row)
                except Exception as e:
                    # Keep going; record a stub row for visibility
                    print(f"[{i}/{len(specs)}] ERROR: {e}")
                    # Optionally: write an error marker file in that run dir

    # Write CSV
    csv_path = sweep_dir / "pf_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_id","q","R","seed","ticks","layers",
            "n_periods","n_lifetimes","n_sizes","n_atlas_speeds","n_graph_speeds",
            "hill_alpha","hill_k","rho_atlas_graph"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"✓ Wrote {csv_path}")

    # Also compute high-level invariance summary across runs and write JSON
    if runs:
        period_samples = [r.periods for r in runs if r.periods.size]
        speed_samples  = [r.atlas_speeds for r in runs if r.atlas_speeds.size]
        size_samples   = [r.sizes for r in runs if r.sizes.size]
        lifetime_samples = [r.lifetimes for r in runs if r.lifetimes.size]
        atlas_all = (
            np.concatenate([r.atlas_speeds for r in runs if r.atlas_speeds.size], axis=0)
            if any(r.atlas_speeds.size for r in runs) else np.array([])
        )
        graph_all = (
            np.concatenate([r.graph_speeds for r in runs if r.graph_speeds.size], axis=0)
            if any(r.graph_speeds.size for r in runs) else np.array([])
        )

        inv = summarize_invariance(
            period_samples=period_samples or [np.array([1.0])],
            speed_samples=speed_samples or [np.array([1.0])],
            size_samples=size_samples or [np.array([1.0])],
            lifetime_samples=lifetime_samples or [np.array([1.0])],
            atlas_speeds_all=atlas_all,
            graph_speeds_all=graph_all,
        )
        inv_json = {
            "ks_period_norm_max": inv.ks_period_norm_max,
            "ks_speed_norm_max": inv.ks_speed_norm_max,
            "ks_size_norm_max": inv.ks_size_norm_max,
            "hill_alpha_mean": inv.hill_alpha_mean,
            "hill_alpha_std": inv.hill_alpha_std,
            "atlas_graph_spearman": inv.atlas_graph_spearman,
        }
        with (sweep_dir / "pf_invariance_summary.json").open("w", encoding="utf-8") as f:
            json.dump(inv_json, f, indent=2)
        print(f"✓ Wrote {sweep_dir/'pf_invariance_summary.json'}")

    return runs, sweep_dir
