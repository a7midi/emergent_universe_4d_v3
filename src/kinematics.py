import math
from typing import Optional, Dict, Any, Iterable

import numpy as np


def _finite(x: float, default: float = 0.0) -> float:
    """Return x if finite else default (avoids NaN/Inf leaking into logs)."""
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else float(default)


def _as_list_finite(arr: Iterable[float], default: float = 0.0) -> list[float]:
    return [ _finite(v, default) for v in arr ]


def calculate_kinematics(
    particle,              # dataclass-like: has .nodes (iterable[int])
    atlas,                 # provides .id_map[node] -> idx and .global_coords[idx] -> R^4
    metric,                # provides .get_symmetric_radius(center_node:int, node:int) -> float
    last: Optional[Dict[str, Any]] = None,  # last kinematics dict (from previous tick)
    dt: float = 1.0,       # tick-time step (simulation advances one tick per call)
) -> Dict[str, Any]:
    """
    Compute centroid (4-vector), a robust cluster radius, and 3-velocity.

    Velocity is computed in *tick time*: v = Δx / dt using the previous centroid
    passed via `last`. This avoids stalls when τ (layer/time-like coord) does not
    advance by 1 or is ill-behaved for some clusters.

    Returns a dict with keys:
        - 'centroid': [x, y, z, τ]
        - 'radius'  : float
        - 'velocity': [vx, vy, vz]
    All values are guaranteed finite.
    """
    # Empty cluster → zeroed observables
    if not getattr(particle, "nodes", None):
        return {"centroid": [0.0, 0.0, 0.0, 0.0], "radius": 0.0, "velocity": [0.0, 0.0, 0.0]}

    # Map nodes to atlas indices; drop nodes missing from atlas (should be rare)
    nodes = list(particle.nodes)
    idxs = [atlas.id_map[n] for n in nodes if n in atlas.id_map]
    if not idxs:
        return {"centroid": [0.0, 0.0, 0.0, 0.0], "radius": 0.0, "velocity": [0.0, 0.0, 0.0]}

    # Pull coordinates and drop any rows containing NaNs
    P = atlas.global_coords[idxs]
    if not isinstance(P, np.ndarray):
        P = np.asarray(P, dtype=float)
    P = P[~np.isnan(P).any(axis=1)]
    if P.size == 0:
        return {"centroid": [0.0, 0.0, 0.0, 0.0], "radius": 0.0, "velocity": [0.0, 0.0, 0.0]}

    # Centroid in R^4
    centroid = P.mean(axis=0)

    # Radius: choose the node nearest to centroid (in x,y,z) as a robust interior
    # reference; then take the maximum finite symmetric radius within the cluster.
    try:
        # nearest node index in our local array
        nearest_local = int(np.argmin(np.sum((P[:, :3] - centroid[:3]) ** 2, axis=1)))
        center_node = nodes[nearest_local]
    except Exception:
        # fallback if anything odd happens
        center_node = nodes[0]

    radii = []
    for n in nodes:
        r = metric.get_symmetric_radius(center_node, n)
        if math.isfinite(r):
            radii.append(float(r))
    radius = max(radii) if radii else 0.0

    # Velocity in tick-time
    vel = np.zeros(3, dtype=float)
    if last is not None and isinstance(last.get("centroid"), (list, tuple, np.ndarray)) and dt:
        prev_xyz = np.array(last["centroid"][:3], dtype=float)
        cur_xyz  = centroid[:3].astype(float, copy=False)
        diff = cur_xyz - prev_xyz
        # divide by dt (usually 1 tick) and clamp to finite values
        with np.errstate(all="ignore"):
            v = diff / float(dt)
        if np.all(np.isfinite(v)):
            vel = v
        else:
            vel = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "centroid": _as_list_finite(centroid.tolist()),
        "radius": _finite(radius),
        "velocity": _as_list_finite(vel.tolist()),
    }
