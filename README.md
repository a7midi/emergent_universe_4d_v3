# Emergent-Universe Simulation Suite

This is a **computational laboratory** for exploring the *Deterministic Causal-Site* programme.

The three companion research papers prove (mathematically) that **spacetime** and stable, **particle-like structures** can emerge from nothing more than a deterministic network of cause-and-effect relationships.  
This software lets you **build that universe**, **run it forward in time**, and **watch it evolve**—so you can see the theory’s predictions play out.

---

## 1. How it works – in plain language

Think of the program as a “universe engine” that runs in three main stages:

1. **Build a world of causes and effects**  
   We create a large layered network where arrows point from earlier events to later ones. This is called a **finite acyclic causal site**—a fancy name for “a one-way web of influence” where cycles are impossible.

2. **Give every event a simple ‘state’ and evolve it deterministically**  
   Each node (event) carries a small integer “tag.” Every tick, it updates its tag based only on the tags of its parent events, using a fixed rule (**fusion mode**). This step is completely deterministic—no randomness after the graph is built.

3. **Detect little machines inside the chaos**  
   We watch the evolving tags to find **clusters of nodes** that stay together, repeat their patterns in time, and don’t decompose into smaller repeating parts. These are our “particles.” We also estimate each node’s position in space by looking at how it’s connected locally and stitching these local maps together, so we can plot particle **world-lines** in an emergent spacetime.

---

## 2. Core architecture

**Phase 1 – Emergent geometry**

- `causal_site.py` – builds the layered graph (the causal site).
- `depth_metric.py` – computes a fast reachability-based “distance” from one event to another.
- `chart_atlas.py` – finds small local maps (charts) using hop-distance, embeds them in 3-D, and stitches them into a global coordinate system `(x, y, z, τ)` where `τ` is the layer (time).

**Phase 2 – Simulation & detection**

- `state_manager.py` – updates all node tags each tick using the chosen **fusion mode** (`injective`, `sum_mod_q`, `quadratic`…).
- `particle_detector.py` – detects repeating, bounded, indecomposable clusters of nodes.
- `kinematics.py` – calculates a particle’s centroid, radius, and 3-velocity in atlas coordinates.

**Outputs**

- `results/static_universe.json` – the fixed graph and each node’s 4-D coordinates.
- `results/simulation_log.jsonl` – a log of every tick: which particles exist, their properties, and summary stats.

---

## 3. Module-by-module

| File                  | Role                                           | In plain words |
|-----------------------|------------------------------------------------|----------------|
| **`causal_site.py`**  | Graph substrate constructor                     | Builds the universe’s “cause-and-effect” network and links in the geometry tools. |
| **`depth_metric.py`** | Quasi-metric calculator                         | Fast look-up for “how far in time” one event is from another (if reachable). |
| **`chart_atlas.py`**  | 4-D spacetime atlas builder                     | Figures out where each node should go in space/time by combining small local maps. |
| **`state_manager.py`**| Tag update engine                               | Advances the universe’s state one tick at a time using your chosen rule. |
| **`particle_detector.py`** | Particle finder                           | Scans for stable, repeating groups of nodes and assigns them IDs. |
| **`kinematics.py`**   | Motion analyser                                 | Measures each particle’s position, size, and speed (in the emergent coordinates). |
| **`export_data.py`**  | Orchestrator                                    | Runs the build → simulate → detect pipeline and writes the results to disk. |
| **`visualizer.html`** | 3-D viewer                                      | Lets you fly around the universe, see the static network, and watch particles move. |
| **`infographic.html`**| Stats dashboard                                 | Shows particle counts, periods, lifetimes, sizes, and speeds as charts. |
| **`build_scene.py`**  | GLTF exporter                                   | Optional: saves the static network as a `.glb` file for Blender/Unity/etc. |

---

## 4. Installation & quick start

1. Create and activate a Python virtual environment, then install dependencies:

```bash
cd <project-folder>
python -m venv venv
venv\Scripts\activate   # on Windows
source venv/bin/activate # on macOS/Linux
pip install -r requirements.txt
requirements.txt includes:
numpy, networkx, tqdm, pyyaml, scikit-learn, scipy, pygltflib.

Edit config.yaml to set:

Graph size and density

Update rule (fusion_mode)

Particle detection thresholds

Run a full simulation:

bash
Copy
Edit
python export_data.py
You’ll see a “Stitching Charts” progress bar while geometry is built, then tick progress as the simulation runs.

Visualise the results:

bash
Copy
Edit
python -m http.server
Open in your browser:

http://localhost:8000/visualizer.html – fly through 3-D spacetime, see particles move.

http://localhost:8000/infographic.html – drag results/simulation_log.jsonl in to see charts.

5. Understanding the output
Visualizer

Grey graph = the static causal network.

Coloured spheres = detected particles, size = radius, colour = oscillation period.

HUD buttons: pause, scrub, hide substrate, hide world-lines.

Dashboard KPIs

Total particles – number of distinct clusters detected.

Unique periods – how many different loop periods exist.

Longest lifetime – ticks from first to last detection of the longest-lived particle.

Total ticks – how long the simulation ran.

Charts

Doughnut: particle counts per period.

Bar: average lifetime per period.

Scatter: speed vs cluster size.

Histogram: lifetime distribution.

If the scatter plot is empty or lifetimes ≤ 3, the “universe” is cold (low activity). Increase max_out_degree_R and/or run for more ticks to “warm it up” and see long-lived movers.

6. The big picture
This program isn’t hard-coding space, time, or particles. It starts with only:

A causal network (who can influence whom)

A deterministic update rule

From that, it derives:

Geometry – how to place events in an emergent space/time

Motion – how patterns move through that space

Structure – which patterns are stable enough to call “particles”

You can change the rules and watch a completely different emergent world unfold.

Happy experimenting!

pgsql
Copy
Edit

Do you want me to also create a **diagram** in Markdown/ASCII for this README that visually shows  
`Build causal site → Assign tags → Evolve → Detect particles → Visualize`? It would make it even more beginner-friendly.






