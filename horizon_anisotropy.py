#!/usr/bin/env python
"""
horizon_anisotropy.py
─────────────────────
Post‑simulation geometry probe:
    σ_d  = variance of radial distance within each depth‑layer d.

Usage
-----
    python horizon_anisotropy.py  path/to/static_universe.json
    python horizon_anisotropy.py  static_universe.json  --save-plot anisotropy.png
"""

from __future__ import annotations
import json, math, argparse, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_nodes(path: Path):
    with path.open() as f:
        raw = json.load(f)
    by_depth = defaultdict(list)
    for meta in raw["nodes"].values():
        layer = int(meta["layer"])
        x, y, z = meta["position"][:3]
        by_depth[layer].append((x, y, z))
    return by_depth


def anisotropy_per_depth(grouped):
    """
    Returns two equal‑length arrays: depths, sigma.
    σ_d is 0 if there are < 2 nodes in that depth.
    """
    depths, sigmas = [], []
    for d in sorted(grouped):
        P = np.array(grouped[d], dtype=float)
        if len(P) < 2:
            depths.append(d)
            sigmas.append(0.0)
            continue
        centroid = P.mean(axis=0)
        radii = np.linalg.norm(P - centroid, axis=1)
        sigma = radii.var()
        depths.append(d)
        sigmas.append(float(sigma))
    return np.array(depths), np.array(sigmas)


def make_plot(depths, sigmas, out: Path | None):
    plt.figure(figsize=(6, 4))
    plt.plot(depths, sigmas, marker="o", linewidth=1.2)
    plt.xlabel("Depth d")
    plt.ylabel(r"$\sigma_d$  (radial variance)")
    plt.title("Horizon anisotropy vs. depth")
    plt.grid(alpha=0.3)
    if out:
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        print(f"✓ plot saved to {out}")
    else:
        plt.show()


def main(argv=None):
    p = argparse.ArgumentParser(description="Horizon anisotropy analysis")
    p.add_argument("json", help="static_universe.json produced by export_data.py")
    p.add_argument("--save-plot", metavar="PNG", help="save matplotlib figure")
    args = p.parse_args(argv)

    path = Path(args.json)
    if not path.exists():
        sys.exit(f"Error: {path} not found")

    grouped = load_nodes(path)
    depths, sigmas = anisotropy_per_depth(grouped)

    # console table
    print("\nDepth   σ_d (variance)")
    print("-----------------------")
    for d, s in zip(depths, sigmas):
        print(f"{d:>5}   {s:.6e}")

    # optional plot
    make_plot(depths, sigmas, Path(args.save_plot) if args.save_plot else None)


if __name__ == "__main__":
    main()
