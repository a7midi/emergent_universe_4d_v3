#!/usr/bin/env python
"""
pf_run.py — CLI wrapper for running parameter-free sweeps.

Examples:
  # Small sanity sweep
  python pf_run.py --q 13 17 --R 4 5 --seeds 42 --ticks 800 --jobs 4

  # Larger overnight sweep
  python pf_run.py --q 11 13 17 19 --R 3 4 5 --seeds 17 42 73 --ticks 12000 --layers 140 --jobs 10
"""

from __future__ import annotations
import argparse
from pathlib import Path

from src.pf_demo.sweep import run_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a parameter-free sweep using src.pf_demo.sweep.run_grid"
    )
    # Required grids
    parser.add_argument("--q", type=int, nargs="+", required=True,
                        help="List of q values to sweep over (e.g. 11 13 17 19)")
    parser.add_argument("--R", type=int, nargs="+", required=True,
                        help="List of R values to sweep over (e.g. 3 4 5)")

    # Common run controls
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="List of random seeds (default: 42)")
    parser.add_argument("--ticks", type=int, default=8000,
                        help="Number of simulation ticks per run (default: 8000)")
    parser.add_argument("--layers", type=int, default=140,
                        help="Number of causal-site layers (default: 140)")
    parser.add_argument("--chart_scale_k", type=int, default=4,
                        help="Chart scale k for atlas building (default: 4)")

    # Optional causal-site overrides (leave unset to use base config.yaml)
    parser.add_argument("--avg_nodes_per_layer", type=int,
                        help="Override avg_nodes_per_layer in causal_site (optional)")
    parser.add_argument("--edge_probability", type=float,
                        help="Override edge_probability in causal_site (optional)")
    parser.add_argument("--max_lookback_layers", type=int,
                        help="Override max_lookback_layers in causal_site (optional)")

    # Infra / paths
    parser.add_argument("--project_root", type=str, default=".",
                        help="Path to project root containing export_data.py and config.yaml (default: .)")
    parser.add_argument("--out_root", type=str,
                        help="Explicit output directory for sweep results (default: results/pf_runs/<timestamp>)")
    parser.add_argument("--python_exe", type=str,
                        help="Python executable to use for per-run subprocesses (default: current interpreter)")

    # Parallelism
    parser.add_argument("--jobs", type=int, default=1,
                        help="How many runs to execute in parallel (default: 1)")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    print(f"Running sweep in: {project_root}")
    if args.jobs and args.jobs > 1:
        print(f"Parallel jobs: {args.jobs}")

    runs, sweep_dir = run_grid(
        q_list=args.q,
        R_list=args.R,
        seeds=args.seeds,
        ticks=args.ticks,
        layers=args.layers,
        avg_nodes_per_layer=args.avg_nodes_per_layer,
        edge_probability=args.edge_probability,
        max_lookback_layers=args.max_lookback_layers,
        chart_scale_k=args.chart_scale_k,
        project_root=str(project_root),
        out_root=args.out_root,
        python_exe=args.python_exe,
        max_workers=args.jobs,
    )

    print("\n=== Sweep complete ===")
    print(f"Results folder: {sweep_dir}")
    print(f"Total runs completed: {len(runs)}")


if __name__ == "__main__":
    main()
