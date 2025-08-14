#!/usr/bin/env python3
"""
report.py — Generate a self-contained HTML report for a parameter-free sweep.

Usage:
  python report.py
  python report.py --sweep-dir results/pf_runs/20250809-190401
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any


def _latest_sweep_dir(base: Path) -> Path | None:
    root = base / "results" / "pf_runs"
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def _load_csv(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        return reader.fieldnames or [], rows


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _coerce_numbers(rows: List[Dict[str, Any]]) -> None:
    # Convert obvious numeric columns to numbers so D3 doesn't mis-sort
    numeric_keys = {
        "q","R","seed","ticks","layers",
        "n_periods","n_lifetimes","n_sizes","n_atlas_speeds","n_graph_speeds",
        "hill_alpha","hill_k","rho_atlas_graph"
    }
    for r in rows:
        for k in list(r.keys()):
            if k in numeric_keys:
                try:
                    r[k] = float(r[k])
                except Exception:
                    r[k] = None


def _color_for_badge(score: float, t_good: float, t_ok: float, inverse: bool = False) -> str:
    """
    Return a semantic color based on thresholds.
    If inverse=False: small is good (e.g., KS)  -> green below t_good.
    If inverse=True:  large is good (e.g., rho) -> green above t_good.
    """
    if score is None:
        return "#94a3b8"  # slate (unknown)
    if not inverse:
        if score <= t_good: return "#22c55e"  # green
        if score <= t_ok:   return "#f59e0b"  # amber
        return "#ef4444"                      # red
    else:
        if score >= t_good: return "#22c55e"
        if score >= t_ok:   return "#f59e0b"
        return "#ef4444"


def _badge(label: str, score: float | None, fmt: str, t_good: float, t_ok: float, inverse: bool=False) -> str:
    color = _color_for_badge(score if score is not None else float("nan"), t_good, t_ok, inverse=inverse)
    text = ("—" if score is None else (fmt.format(score)))
    return f'<span class="badge" style="background:{color}">{label}: {text}</span>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=str, help="Path to a sweep folder (results/pf_runs/<timestamp>)")
    args = ap.parse_args()

    project_root = Path(".").resolve()
    sweep_dir = Path(args.sweep_dir).resolve() if args.sweep_dir else _latest_sweep_dir(project_root)
    if not sweep_dir or not sweep_dir.exists():
        print("No sweep folder found. Run a sweep first or pass --sweep-dir.")
        return

    csv_path = sweep_dir / "pf_runs.csv"
    inv_path = sweep_dir / "pf_invariance_summary.json"

    headers, rows = _load_csv(csv_path)
    _coerce_numbers(rows)
    inv = _load_json(inv_path)

    # Headline metrics (may be missing if first sweep had too little data)
    ksP = inv.get("ks_period_norm_max", None)
    ksS = inv.get("ks_speed_norm_max", None)
    ksZ = inv.get("ks_size_norm_max", None)
    hill_mean = inv.get("hill_alpha_mean", None)
    hill_std  = inv.get("hill_alpha_std", None)
    rho = inv.get("atlas_graph_spearman", None)

    # Policy thresholds (tune if you like)
    # For KS (smaller is better): green ≤ 0.10, amber ≤ 0.20, else red.
    # For Spearman rho (bigger is better): green ≥ 0.80, amber ≥ 0.60, else red.
    # For Hill mean (we just display; stability judged by std): small std is good.
    ks_badges = (
        _badge("KS period (norm)", ksP, "{:.3f}", 0.10, 0.20, inverse=False),
        _badge("KS speed (norm)",  ksS, "{:.3f}", 0.10, 0.20, inverse=False),
        _badge("KS size  (norm)",  ksZ, "{:.3f}", 0.10, 0.20, inverse=False),
    )
    rho_badge = _badge("Atlas↔Graph Spearman", rho, "{:.3f}", 0.80, 0.60, inverse=True)

    # Embed data as JSON in the HTML
    data_json = json.dumps(rows, ensure_ascii=False)
    inv_json = json.dumps(inv, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Parameter-Free Sweep Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root {{
    --bg:#0f172a; --fg:#e2e8f0; --muted:#94a3b8; --panel:#111827; --grid:#1f2937;
    --accent:#38bdf8; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444;
  }}
  html,body{{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Roboto,sans-serif}}
  header{{padding:16px 20px;border-bottom:1px solid #1f2937}}
  h1{{margin:0;font-size:20px}}
  .sub{{color:var(--muted);font-size:12px}}
  main{{display:grid;grid-template-columns:1fr;gap:16px;padding:16px 20px}}
  @media(min-width:1100px){{ main{{grid-template-columns:1.2fr .8fr}} }}
  section{{background:var(--panel);border:1px solid #1f2937;border-radius:10px;padding:12px 14px}}
  h2{{margin:0 0 8px 0;font-size:16px}}
  .badges{{display:flex;flex-wrap:wrap;gap:8px}}
  .badge{{display:inline-block;padding:4px 8px;border-radius:999px;color:#0b1220;font-weight:600;font-size:12px}}
  .small{{font-size:12px;color:var(--muted)}}
  #heatmap svg, #scatter svg, #hist svg {{width:100%;height:400px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th,td{{padding:6px 8px;border-bottom:1px solid #1f2937;text-align:left;white-space:nowrap}}
  th{{position:sticky;top:0;background:#0b1220;z-index:1}}
  .pill{{padding:2px 6px;border-radius:999px;background:#1f2937;color:var(--fg);font-size:12px}}
  .legend{{display:flex;gap:10px;align-items:center;font-size:12px;color:var(--muted)}}
</style>
</head>
<body>
<header>
  <h1>Parameter-Free Sweep Report</h1>
  <div class="sub">{sweep_dir.as_posix()}</div>
</header>

<main>
  <section>
    <h2>Headlines</h2>
    <div class="badges">
      {ks_badges[0]} {ks_badges[1]} {ks_badges[2]} {rho_badge}
      <span class="badge" style="background:#38bdf8">Hill α mean: {"—" if hill_mean is None else f"{hill_mean:.3f}"}</span>
      <span class="badge" style="background:#38bdf8">Hill α std: {"—" if hill_std  is None else f"{hill_std:.3f}"}</span>
    </div>
    <p class="small" style="margin-top:6px">
      KS thresholds: green ≤ 0.10, amber ≤ 0.20. Spearman thresholds: green ≥ 0.80, amber ≥ 0.60.
      The Hill tail index reflects heavy-tail behavior in lifetimes; a small std across runs indicates stability.
    </p>
  </section>

  <section id="heatmap">
    <h2>(q,R) heatmap — detections per 1k ticks</h2>
    <div class="legend">
      <span>Each cell: average <span class="pill">n_lifetimes / (ticks/1000)</span> across seeds</span>
    </div>
    <svg></svg>
  </section>

  <section id="scatter">
    <h2>Hill tail index per run</h2>
    <svg></svg>
    <p class="small">Only runs with enough lifetimes produce a finite tail index. Expect more points with longer ticks.</p>
  </section>

  <section id="hist">
    <h2>Per-run summary (periods / sizes)</h2>
    <svg></svg>
    <p class="small">Bars show normalized counts. Click legend to toggle series.</p>
  </section>

  <section id="table">
    <h2>All runs</h2>
    <div class="small" style="margin-bottom:6px">Click headers to sort.</div>
    <div style="max-height:460px;overflow:auto;border:1px solid #1f2937;border-radius:8px">
      <table id="runs">
        <thead></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>
</main>

<script id="runs-json" type="application/json">{data_json}</script>
<script id="inv-json"  type="application/json">{inv_json}</script>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
const runs = JSON.parse(document.getElementById('runs-json').textContent || '[]');
const inv  = JSON.parse(document.getElementById('inv-json').textContent || '{{}}');

// Coerce numeric fields (defensive in case CSV had strings)
const numKeys = new Set(["q","R","seed","ticks","layers",
  "n_periods","n_lifetimes","n_sizes","n_atlas_speeds","n_graph_speeds",
  "hill_alpha","hill_k","rho_atlas_graph"]);
for(const r of runs){ for(const k of Object.keys(r)){ if(numKeys.has(k)) r[k]=+r[k]; } }

// ---- Heatmap (detections per 1k ticks) ---------------------------------
(function(){
  const svg = d3.select("#heatmap svg");
  const box = svg.node().getBoundingClientRect();
  const W = box.width, H = box.height, m = {t:20,r:10,b:40,l:40};
  svg.selectAll("*").remove();

  const qVals = Array.from(new Set(runs.map(d=>d.q))).sort((a,b)=>a-b);
  const RVals = Array.from(new Set(runs.map(d=>d.R))).sort((a,b)=>a-b);

  // Aggregate by (q,R): avg detections per 1k ticks, where we use n_lifetimes as proxy for “detections”
  const key = (q,R)=>q+"_"+R;
  const acc = new Map();
  for(const r of runs){
    const k = key(r.q,r.R);
    if(!acc.has(k)) acc.set(k, {q:r.q,R:r.R, sum:0, cnt:0, ticks:r.ticks||1000});
    const a = acc.get(k);
    const det_per_1k = (r.n_lifetimes || 0) / ((r.ticks||1000)/1000.0);
    a.sum += det_per_1k; a.cnt += 1;
  }
  const cells = Array.from(acc.values()).map(a=>({q:a.q,R:a.R,val:a.cnt? a.sum/a.cnt : 0}));

  const x = d3.scaleBand().domain(qVals).range([m.l, W-m.r]).padding(0.06);
  const y = d3.scaleBand().domain(RVals).range([H-m.b, m.t]).padding(0.06);
  const maxV = d3.max(cells, d=>d.val)||1;
  const c = d3.scaleSequential(d3.interpolateTurbo).domain([0,maxV]);

  svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x));
  svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y));

  svg.selectAll("rect.cell").data(cells).join("rect")
    .attr("class","cell")
    .attr("x",d=>x(d.q)).attr("y",d=>y(d.R))
    .attr("width",x.bandwidth()).attr("height",y.bandwidth())
    .attr("rx",6)
    .attr("fill",d=>c(d.val))
    .append("title").text(d=>`q=${d.q}, R=${d.R}\navg detections/1k ticks: ${d.val.toFixed(2)}`);
})();

// ---- Scatter: Hill tail index per run -----------------------------------
(function(){
  const svg = d3.select("#scatter svg");
  const box = svg.node().getBoundingClientRect();
  const W = box.width, H = box.height, m = {t:20,r:20,b:40,l:50};
  svg.selectAll("*").remove();

  const data = runs.filter(d=>isFinite(d.hill_alpha) && d.hill_alpha>0);
  if(!data.length){
    svg.append("text").attr("x",W/2).attr("y",H/2).attr("text-anchor","middle").text("No finite Hill estimates yet (need longer runs)");
    return;
  }
  const x = d3.scalePoint().domain(data.map(d=>d.run_id)).range([m.l, W-m.r]).padding(0.5);
  const y = d3.scaleLinear().domain(d3.extent(data, d=>d.hill_alpha)).nice().range([H-m.b, m.t]);

  svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).tickFormat(d=>d.replace(/_seed.*/,"")).tickSizeOuter(0)).selectAll("text").attr("transform","rotate(-30)").style("text-anchor","end");
  svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y));

  svg.selectAll("circle").data(data).join("circle")
    .attr("cx",d=>x(d.run_id)).attr("cy",d=>y(d.hill_alpha))
    .attr("r",5).attr("fill","#38bdf8").attr("opacity",0.9)
    .append("title").text(d=>`${d.run_id}\nHill α: ${d.hill_alpha.toFixed(3)} (k=${d.hill_k|0})`);
})();

// ---- Mini histograms: periods & sizes (per run) -------------------------
(function(){
  const svg = d3.select("#hist svg");
  const box = svg.node().getBoundingClientRect();
  const W = box.width, H = box.height, m = {t:20,r:10,b:40,l:50};
  svg.selectAll("*").remove();

  // We don't have per-run distributions here, only counts. So we show bars by run:
  const series = [
    {key:"n_periods", label:"period detections", color:"#22c55e"},
    {key:"n_sizes",   label:"cluster sizes (counted)", color:"#a78bfa"}
  ];
  let visible = new Set(series.map(s=>s.key));

  const runsSorted = runs.slice().sort((a,b)=> (a.q-b.q) || (a.R-b.R) || (a.seed-b.seed));
  const x = d3.scaleBand().domain(runsSorted.map(d=>d.run_id)).range([m.l, W-m.r]).padding(0.2);
  const y = d3.scaleLinear().domain([0, d3.max(runsSorted, d=>Math.max(d.n_periods||0, d.n_sizes||0)) || 1]).nice().range([H-m.b, m.t]);

  svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).tickFormat(d=>d.replace(/_seed.*/,"")).tickSizeOuter(0)).selectAll("text").attr("transform","rotate(-30)").style("text-anchor","end");
  svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y));

  function drawBars(){
    svg.selectAll("g.series").remove();
    series.filter(s=>visible.has(s.key)).forEach((s,si)=>{
      const g = svg.append("g").attr("class","series");
      const bw = x.bandwidth() / (visible.size||1);
      let offset = Array.from(visible).indexOf(s.key);
      g.selectAll("rect").data(runsSorted).join("rect")
        .attr("x",d=>x(d.run_id)+offset*bw)
        .attr("y",d=>y(+d[s.key]||0))
        .attr("width",bw)
        .attr("height",d=>y(0)-y(+d[s.key]||0))
        .attr("fill",s.color)
        .append("title").text(d=>`${s.label} — ${d.run_id}: ${(+d[s.key]||0)}`);
    });
  }
  drawBars();

  // Legend toggles
  const legend = d3.select("#hist").append("div").style("display","flex").style("gap","10px").style("marginTop","6px");
  series.forEach(s=>{
    const btn = legend.append("button").text(s.label)
      .style("background", s.color).style("border","none").style("color","#0b1220")
      .style("padding","4px 8px").style("borderRadius","999px").style("cursor","pointer").style("fontWeight","700");
    btn.on("click", ()=>{
      if(visible.has(s.key)) visible.delete(s.key); else visible.add(s.key);
      drawBars();
      btn.style("opacity", visible.has(s.key)? 1 : 0.35);
    });
  });
})();

// ---- Table --------------------------------------------------------------
(function(){
  const table = d3.select("#runs");
  const columns = [
    "run_id","q","R","seed","ticks","layers",
    "n_periods","n_lifetimes","n_sizes","n_atlas_speeds","n_graph_speeds",
    "hill_alpha","hill_k","rho_atlas_graph"
  ];
  const thead = table.select("thead").append("tr");
  columns.forEach(c=> thead.append("th").text(c));

  const tbody = table.select("tbody");
  const fmt = (k,v)=>{
    if(v==null || Number.isNaN(v)) return "—";
    if(["q","R","seed","layers","ticks","n_periods","n_lifetimes","n_sizes","n_atlas_speeds","n_graph_speeds","hill_k"].includes(k)) return String(Math.round(+v));
    if(["hill_alpha","rho_atlas_graph"].includes(k)) return (+v).toFixed(3);
    return String(v);
  };

  let cur = runs.slice();
  function render(){
    tbody.selectAll("tr").remove();
    const tr = tbody.selectAll("tr").data(cur).join("tr");
    columns.forEach(k=>{
      tr.append("td").text(d=>fmt(k,d[k]));
    });
  }
  render();

  // Click-to-sort
  let asc=true, lastKey=null;
  thead.selectAll("th").on("click", function(_,key){
    if(lastKey===key) asc=!asc; else asc=true, lastKey=key;
    cur.sort((a,b)=>{
      const va=a[key], vb=b[key];
      const A = (va==null || Number.isNaN(va)) ? -Infinity : +va;
      const B = (vb==null || Number.isNaN(vb)) ? -Infinity : +vb;
      if(isFinite(A) && isFinite(B)) return asc? A-B : B-A;
      return asc? String(a[key]).localeCompare(String(b[key])) : String(b[key]).localeCompare(String(a[key]));
    });
    render();
  });
})();
</script>

</body>
</html>
"""
    out_path = sweep_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"✓ wrote {out_path}")


if __name__ == "__main__":
    main()
