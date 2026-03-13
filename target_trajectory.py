"""
Target trajectory definitions and reward functions for the Four-Bar Linkage Synthesizer.

Supports: straight_line, semicircle, ellipse, figure8, custom (user-drawn).
"""

import numpy as np


# ── Trajectory type schemas ────────────────────────────────────────────────────

TRAJECTORY_TYPES = [
    {
        "id": "straight_line",
        "label": "Straight Line (y = c)",
        "params": [
            {"key": "c", "label": "y = c (m)", "type": "float",
             "default": 0.0, "min": -0.3, "max": 0.3},
        ],
    },
    {
        "id": "semicircle",
        "label": "Semicircle",
        "params": [
            {"key": "cx",    "label": "Centre x (m)", "type": "float",
             "default": 0.15, "min": -0.3, "max": 0.5},
            {"key": "cy",    "label": "Centre y (m)", "type": "float",
             "default": 0.0,  "min": -0.3, "max": 0.3},
            {"key": "r",     "label": "Radius (m)",   "type": "float",
             "default": 0.08, "min": 0.01, "max": 0.3},
            {"key": "upper", "label": "Upper half",   "type": "bool",
             "default": True},
        ],
    },
    {
        "id": "ellipse",
        "label": "Ellipse",
        "params": [
            {"key": "cx", "label": "Centre x (m)", "type": "float",
             "default": 0.15, "min": -0.3, "max": 0.5},
            {"key": "cy", "label": "Centre y (m)", "type": "float",
             "default": 0.0,  "min": -0.3, "max": 0.3},
            {"key": "rx", "label": "x-radius (m)", "type": "float",
             "default": 0.10, "min": 0.01, "max": 0.3},
            {"key": "ry", "label": "y-radius (m)", "type": "float",
             "default": 0.05, "min": 0.01, "max": 0.3},
        ],
    },
    {
        "id": "figure8",
        "label": "Figure-8",
        "params": [
            {"key": "cx", "label": "Centre x (m)", "type": "float",
             "default": 0.15, "min": -0.3, "max": 0.5},
            {"key": "cy", "label": "Centre y (m)", "type": "float",
             "default": 0.0,  "min": -0.3, "max": 0.3},
            {"key": "a",  "label": "Half-width (m)", "type": "float",
             "default": 0.08, "min": 0.01, "max": 0.3},
        ],
    },
    {
        "id": "custom",
        "label": "Custom Path (Draw)",
        "params": [
            {"key": "points", "label": "Drawn points", "type": "points",
             "default": []},
        ],
    },
]


# ── Public API ─────────────────────────────────────────────────────────────────

def get_trajectory_types():
    """Return the list of trajectory type schemas."""
    return TRAJECTORY_TYPES


def generate_trajectory(traj_type, traj_params, n=200):
    """
    Generate an (n, 2) NumPy array of (x, y) target points.

    Parameters
    ----------
    traj_type   : str   — one of 'straight_line','semicircle','ellipse',
                          'figure8','custom'
    traj_params : dict  — parameter values matching the schema above
    n           : int   — number of points to sample (ignored for 'custom')

    Returns
    -------
    pts : np.ndarray, shape (n, 2) or (len(custom_pts), 2).
          May be (0, 2) for invalid custom paths.
    """
    if traj_type == "straight_line":
        c  = float(traj_params.get("c", 0.0))
        xs = np.linspace(-0.1, 0.4, n)
        return np.column_stack([xs, np.full(n, c)])

    elif traj_type == "semicircle":
        cx    = float(traj_params.get("cx", 0.15))
        cy    = float(traj_params.get("cy", 0.0))
        r     = float(traj_params.get("r",  0.08))
        upper = bool(traj_params.get("upper", True))
        t     = np.linspace(0.0, np.pi, n)
        sign  = 1.0 if upper else -1.0
        return np.column_stack([cx + r * np.cos(t), cy + sign * r * np.sin(t)])

    elif traj_type == "ellipse":
        cx = float(traj_params.get("cx", 0.15))
        cy = float(traj_params.get("cy", 0.0))
        rx = float(traj_params.get("rx", 0.10))
        ry = float(traj_params.get("ry", 0.05))
        t  = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])

    elif traj_type == "figure8":
        cx = float(traj_params.get("cx", 0.15))
        cy = float(traj_params.get("cy", 0.0))
        a  = float(traj_params.get("a",  0.08))
        t  = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        # Lemniscate-style figure-8: x = a*cos(t), y = (a/2)*sin(2t)
        return np.column_stack([cx + a * np.cos(t), cy + (a / 2.0) * np.sin(2.0 * t)])

    elif traj_type == "custom":
        pts = traj_params.get("points", [])
        if len(pts) < 2:
            return np.empty((0, 2))
        return np.array(pts, dtype=float)

    else:
        # Fallback: horizontal line at y = 0
        xs = np.linspace(-0.1, 0.4, n)
        return np.column_stack([xs, np.zeros(n)])


# ── Nearest-neighbour distance metrics ────────────────────────────────────────

def nearest_neighbour_errors(coupler_pts, target_pts):
    """
    Compute nearest-neighbour (NN) distances from each coupler point to the
    closest point on the target curve.

    Parameters
    ----------
    coupler_pts : array-like (N, 2)
    target_pts  : array-like (M, 2)

    Returns
    -------
    mse     : float — mean squared NN distance (m²)
    max_dev : float — maximum NN distance (m)
    """
    cp = np.asarray(coupler_pts, dtype=float)
    tp = np.asarray(target_pts,  dtype=float)
    if cp.shape[0] == 0 or tp.shape[0] == 0:
        return np.inf, np.inf
    # (N, 1, 2) − (1, M, 2)  →  (N, M)
    diff      = cp[:, np.newaxis, :] - tp[np.newaxis, :, :]
    min_dists = np.sqrt(np.sum(diff ** 2, axis=2)).min(axis=1)   # (N,)
    return float(np.mean(min_dists ** 2)), float(np.max(min_dists))


# ── Scalar reward for a single design ─────────────────────────────────────────

def trajectory_path_reward(p, target_pts, enforce_grashof=None):
    """
    Compute reward for design vector *p* against an arbitrary target curve.

    Uses nearest-neighbour distance (not line-fit residuals), so it works
    for any target shape.

    Parameters
    ----------
    p               : array-like (6,) — [L1, L2, L3, L4, xO2, yO2]
    target_pts      : array-like (M, 2)
    enforce_grashof : str or None — 'crank_rocker' to apply Grashof check

    Returns
    -------
    reward      : float
    coupler_pts : np.ndarray (K, 2)   — may be empty if infeasible
    mse         : float  (m²)
    max_dev     : float  (m)
    """
    from path_kinematics import (
        coupler_path, THETA2,
        is_crank_rocker_with_L2_crank, _transmission_penalty,
    )
    p = np.array(p, dtype=float)
    if np.any(p[:4] <= 0):
        return -1e6, np.empty((0, 2)), np.inf, np.inf
    if enforce_grashof == "crank_rocker":
        if not is_crank_rocker_with_L2_crank(*p[:4]):
            return -1e6, np.empty((0, 2)), np.inf, np.inf

    pts, valid_frac, all_valid = coupler_path(p)
    if not all_valid or valid_frac < 0.99 or len(pts) < len(THETA2) * 0.99:
        return -1e6, np.empty((0, 2)), np.inf, np.inf

    tp = np.asarray(target_pts, dtype=float)
    if tp.shape[0] < 2:
        return -1e6, pts, np.inf, np.inf

    mse, max_dev = nearest_neighbour_errors(pts, tp)
    if not np.isfinite(mse):
        return -1e6, pts, np.inf, np.inf

    # Scale NN distances to a comparable range as the straight-line reward.
    # Transmission penalty ensures mechanical feasibility is rewarded.
    pen    = _transmission_penalty(p)
    reward = -(mse * 1000.0 + max_dev * 20.0 + 0.05 * pen)
    return float(reward), pts, float(mse), float(max_dev)


# ── Batch reward for optimisers ────────────────────────────────────────────────

def batch_trajectory_reward(batch_p, target_pts):
    """
    Batch nearest-neighbour reward for arbitrary target trajectories.

    Parameters
    ----------
    batch_p    : array-like (N, 6)
    target_pts : array-like (M, 2)

    Returns
    -------
    rewards : np.ndarray (N,)
    """
    from path_kinematics import (
        coupler_path, THETA2, is_crank_rocker_with_L2_crank,
    )
    batch_p    = np.asarray(batch_p,    dtype=float)
    target_pts = np.asarray(target_pts, dtype=float)
    N          = batch_p.shape[0]
    rewards    = np.full(N, -1e6, dtype=float)

    for i in range(N):
        p = batch_p[i]
        if np.any(p[:4] <= 0):
            continue
        if not is_crank_rocker_with_L2_crank(*p[:4]):
            continue
        pts, valid_frac, all_valid = coupler_path(p)
        if not all_valid or valid_frac < 0.99 or len(pts) < len(THETA2) * 0.99:
            continue
        mse, max_dev = nearest_neighbour_errors(pts, target_pts)
        if not np.isfinite(mse):
            continue
        rewards[i] = -(mse * 1000.0 + max_dev * 20.0)

    return rewards
