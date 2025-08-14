#!/usr/bin/env python3
"""
geometry_report.py
──────────────────
Comprehensive static-geometry probe for the Emergent-Universe suite.

CLI mode (unchanged):
  Calculates per-depth shell statistics from results/static_universe.json and
  saves plots + optional HTML.

Library mode (new hooks used by the simulator/analytics):
  - curvature_proxy(site) -> float
      A lightweight global curvature proxy for the current site.
  - curvature_per_node(site, k) -> np.ndarray (shape: [N])
      Per-node curvature proxy at (coarse) scale k.

Current implementation uses a fast, degree-based proxy:
    κ(v) = R_cap − out_degree(v)
This is deterministic, cheap, and stable. You can later replace these hooks
with an atlas/metric-based estimate without touching the rest of the suite.

Outputs (CLI):
    results/anisotropy.png
    results/mean_radius.png
    results/density.png
    results/diameter.png
    results/gh_drift.png
    results/geometry_report.html   (optional)

Usage:
    python geometry_report.py  results/static_universe.json  [--html report.html]
"""

from __future__ import annotations
import json, math, argparse, sys, pathlib
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ──────────────────────────────────────────────────────────────────────────
# Library hooks (used by export_data.py / rg_analysis.py)
# ──────────────────────────────────────────────────────────────────────────

def _degree_curvature_proxy(graph: nx.DiGraph, R_cap: int | None) -> np.ndarray:
    """Per-node proxy: κ(v) = R_cap − out_degree(v)."""
    nodes = list(graph.nodes)
    if not nodes:
        return np.zeros(0, dtype=float)
    out_deg = np.array([graph.out_degree(n) for n in nodes], dtype=float)
    cap = float(R_cap) if R_cap is not None else (float(np.max(out_deg)) if len(out_deg) else 0.0)
    return (cap - out_deg).astype(float)


def curvature_per_node(site, k: int) -> np.ndarray:
    """
    Per-node curvature at coarse scale k (signature expected by rg_analysis.py).
    For now this returns the degree-based proxy, independent of k.
    You can later incorporate k via hop-ball averaging or atlas-based estimates.
    """
    graph: nx.DiGraph = site.graph
    R_cap = site.config.get("tags", {}).get("max_out_degree_R")
    return _degree_curvature_proxy(graph, R_cap)


def curvature_proxy(site) -> float:
    """
    Global curvature proxy for the current site. Returns the mean of κ(v)
    over all nodes (finite float). Used by export_data.py per-tick summary.
    """
    kappa = curvature_per_node(site, k=getattr(site.config.get("geometry", {}), "get", lambda *_: 0)("chart_scale_k", 0) or 0)
    if kappa.size == 0:
        return float("nan")
    val = float(np.mean(kappa))
    return val if math.isfinite(val) else float("nan")


# ──────────────────────────────────────────────────────────────────────────
# CLI helpers (original report functionality)
# ──────────────────────────────────────────────────────────────────────────

def load_static(path: pathlib.Path):
    with path.open() as f:
        raw = json.load(f)
    if "nodes" not in raw:
        sys.exit("static_universe.json lacks 'nodes'")
    by_depth = defaultdict(list)
    for nid, meta in raw["nodes"].items():
        d = int(meta["layer"])
        x, y, z = meta["position"][:3]
        by_depth[d].append((int(nid), np.array([x, y, z], dtype=float)))
    G_undirected = nx.Graph([(int(u), int(v)) for u, v in raw.get("edges", [])])
    return by_depth, G_undirected


def stats_per_depth(by_depth, G):
    depths, r_mean, sigma, density, diameter, gh = [], [], [], [], [], []

    prev_pts = None
    for d in sorted(by_depth):
        pts = np.array([p for _, p in by_depth[d]], dtype=float)
        N = len(pts)
        centre = pts.mean(axis=0) if N else np.zeros(3, dtype=float)
        radii = np.linalg.norm(pts - centre, axis=1) if N else np.zeros(0, dtype=float)

        depths.append(d)
        r_mean.append(float(np.mean(radii)) if radii.size else 0.0)
        sigma.append(float(np.var(radii)) if radii.size else 0.0)
        denom = max((float(np.mean(radii)) ** 3) if radii.size else 0.0, 1e-9)
        density.append(float(N) / denom)

        # diameter of the largest connected component in this slice
        sub_nodes = [nid for nid, _ in by_depth[d]]
        H = G.subgraph(sub_nodes).copy()
        if len(H) > 1:
            # keep the biggest component only
            largest_cc = max(nx.connected_components(H), key=len)
            Hc = H.subgraph(largest_cc)
            try:
                diam = nx.diameter(Hc)
            except nx.NetworkXError:
                diam = 0
        else:
            diam = 0
        diameter.append(int(diam))

        # GH drift ≈ Procrustes RMS error to previous shell
        if prev_pts is not None and len(pts) >= 3 and len(prev_pts) >= 3:
            A = pts[: min(len(pts), len(prev_pts))]
            B = prev_pts[: len(A)]
            muA, muB = A.mean(0), B.mean(0)
            A0, B0 = A - muA, B - muB
            U, _, Vt = np.linalg.svd(A0.T @ B0)
            R = U @ Vt
            B_rot = B0 @ R
            gh_val = float(np.sqrt(((A0 - B_rot) ** 2).sum() / A0.shape[0]))
            gh.append(gh_val)
        else:
            gh.append(0.0)
        prev_pts = pts
    return depths, r_mean, sigma, density, diameter, gh


def save_plot(x, y, ylabel, outfile, logy=False):
    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(x, y, marker="o", ms=3)
    plt.xlabel("Depth d")
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"✓ {outfile}")


HTML_TEMPLATE = """
<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>Geometry report</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:1rem}
h1{font-size:1.4rem}img{max-width:100%;height:auto;margin-bottom:1rem;border:1px solid #444}
small{color:#888}</style></head><body>
<h1>Emergent-Universe static-geometry report</h1>
<p><small>File: {{ json }}</small></p>
{% for img,title in images %}
<h2>{{ title }}</h2><img src="{{ img }}" alt="{{ title }}"/>
{% endfor %}
</body></html>
"""


def write_html(out_path: pathlib.Path, images: list[tuple[str, str]], json_name: str):
    try:
        from jinja2 import Template
    except ImportError:
        print("Jinja2 missing – HTML report skipped")
        return
    tpl = Template(HTML_TEMPLATE)
    html = tpl.render(images=images, json=json_name)
    out_path.write_text(html)
    print(f"✓ {out_path} (open in browser)")


# ──────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="static_universe.json file")
    ap.add_argument("--max-depth", type=int, help="ignore layers > N")
    ap.add_argument("--html", metavar="FILE", help="save combined HTML report")
    args = ap.parse_args()

    path = pathlib.Path(args.json)
    if not path.exists():
        sys.exit(f"{path} not found")

    by_depth, graph = load_static(path)
    if args.max_depth is not None:
        by_depth = {d: pts for d, pts in by_depth.items() if d <= args.max_depth}

    d, r, sig, rho, diam, gh = stats_per_depth(by_depth, graph)

    # console table
    print(" d   N   r̄      σ        ρ        diam   GH")
    print("---------------------------------------------")
    for i, layer in enumerate(d):
        N = len(by_depth[layer])
        print(f"{layer:2d}  {N:3d}  {r[i]:.3f}  {sig[i]:.3e}  {rho[i]:.3e}  {diam[i]:5d}  {gh[i]:.3e}")

    out_dir = path.parent
    save_plot(d, sig,  "σ_d (variance)",                 out_dir / "anisotropy.png")
    save_plot(d, r,    "mean radius r̄_d",               out_dir / "mean_radius.png")
    save_plot(d, rho,  "node density ρ_d",               out_dir / "density.png", logy=True)
    save_plot(d, diam, "layer causal diameter",          out_dir / "diameter.png")
    save_plot(d, gh,   "GH drift d→d+1  (RMS)",          out_dir / "gh_drift.png")

    if args.html:
        imgs = [("anisotropy.png", "Horizon anisotropy σ_d"),
                ("mean_radius.png", "Mean cosmic radius r̄_d"),
                ("density.png",     "Node density ρ_d"),
                ("diameter.png",    "Largest causal diameter per slice"),
                ("gh_drift.png",    "Gromov–Hausdorff drift")]
        write_html(path.parent / args.html, imgs, path.name)


if __name__ == "__main__":
    main()
