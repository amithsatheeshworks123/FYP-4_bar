"""Target trajectory shapes for four-bar linkage path synthesis."""

import numpy as np
from path_kinematics import coupler_path, is_crank_rocker_with_L2_crank

_2PI = 2 * np.pi


def make_trajectory(traj_type: str, params: dict) -> np.ndarray:
    """Return a (360, 2) array of (x, y) reference points for the given trajectory type.

    Parameters
    ----------
    traj_type : str
        One of 'straight', 'ellipse', 'kidney', 'teardrop', 'circle', 'custom'.
    params : dict
        Override default parameters for the chosen type.

    Returns
    -------
    np.ndarray, shape (360, 2)
    """
    N = 360

    if traj_type == "straight":
        c = params.get("c", 0.0)
        x_min = params.get("x_min", 0.05)
        x_max = params.get("x_max", 0.40)
        x = np.linspace(x_min, x_max, N)
        y = np.full(N, c)
        return np.column_stack([x, y])

    elif traj_type == "ellipse":
        cx = params.get("cx", 0.20)
        cy = params.get("cy", 0.10)
        rx = params.get("rx", 0.12)
        ry = params.get("ry", 0.06)
        t = np.linspace(0, _2PI, N, endpoint=False)
        x = cx + rx * np.cos(t)
        y = cy + ry * np.sin(t)
        return np.column_stack([x, y])

    elif traj_type == "kidney":
        cx = params.get("cx", 0.20)
        cy = params.get("cy", 0.10)
        a = params.get("a", 0.10)
        b = params.get("b", 0.06)
        t = np.linspace(0, _2PI, N, endpoint=False)
        r = a + b * np.cos(t)
        x = cx + r * np.cos(t)
        y = cy + r * np.sin(t)
        return np.column_stack([x, y])

    elif traj_type == "teardrop":
        cx = params.get("cx", 0.20)
        cy = params.get("cy", 0.10)
        rx = params.get("rx", 0.10)
        ry = params.get("ry", 0.08)
        t = np.linspace(0, _2PI, N, endpoint=False)
        x = cx + rx * np.cos(t)
        y = cy + ry * np.sin(t) * np.sin(t / 2)
        return np.column_stack([x, y])

    elif traj_type == "circle":
        cx = params.get("cx", 0.20)
        cy = params.get("cy", 0.10)
        r = params.get("r", 0.08)
        t = np.linspace(0, _2PI, N, endpoint=False)
        x = cx + r * np.cos(t)
        y = cy + r * np.sin(t)
        return np.column_stack([x, y])

    elif traj_type == "custom":
        raw = np.asarray(params.get("points", [[0.0, 0.0], [1.0, 0.0]]), dtype=float)
        if len(raw) < 2:
            raise ValueError("'custom' trajectory requires at least 2 points.")
        # Compute cumulative arc length
        diffs = np.diff(raw, axis=0)
        seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
        cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = cum_len[-1]
        if total_len == 0:
            raise ValueError("'custom' trajectory points are all identical.")
        # Resample uniformly along arc length
        s_uniform = np.linspace(0.0, total_len, N)
        x = np.interp(s_uniform, cum_len, raw[:, 0])
        y = np.interp(s_uniform, cum_len, raw[:, 1])
        return np.column_stack([x, y])

    else:
        raise ValueError(f"Unknown trajectory type: '{traj_type}'. "
                         "Choose from: straight, ellipse, kidney, teardrop, circle, custom.")


def trajectory_reward(coupler_pts: np.ndarray, target_pts: np.ndarray) -> float:
    """Bidirectional nearest-neighbour reward.

    For each coupler point, find distance to nearest target point.
    For each target point, find distance to nearest coupler point.
    Returns negative mean of both sets of distances (0.0 for perfect match).

    Parameters
    ----------
    coupler_pts : np.ndarray, shape (M, 2)
    target_pts  : np.ndarray, shape (K, 2)

    Returns
    -------
    float
        <= 0.0; higher (less negative) is better.
    """
    # coupler -> target: (M, 1, 2) - (1, K, 2) -> (M, K)
    diff_ct = coupler_pts[:, np.newaxis, :] - target_pts[np.newaxis, :, :]
    dist_ct = np.hypot(diff_ct[:, :, 0], diff_ct[:, :, 1])
    min_c2t = dist_ct.min(axis=1)  # (M,)

    # target -> coupler: (K, 1, 2) - (1, M, 2) -> (K, M)
    diff_tc = target_pts[:, np.newaxis, :] - coupler_pts[np.newaxis, :, :]
    dist_tc = np.hypot(diff_tc[:, :, 0], diff_tc[:, :, 1])
    min_t2c = dist_tc.min(axis=1)  # (K,)

    return -float(np.concatenate([min_c2t, min_t2c]).mean())


def batch_trajectory_reward(batch_p: np.ndarray, target_pts: np.ndarray) -> np.ndarray:
    """Compute trajectory rewards for a batch of design vectors.

    Parameters
    ----------
    batch_p    : np.ndarray, shape (N, 6)
        Each row is [L1, L2, L3, L4, xO2, yO2].
    target_pts : np.ndarray, shape (M, 2)

    Returns
    -------
    np.ndarray, shape (N,)
        Reward per design; -1e6 for infeasible designs.
    """
    N = len(batch_p)
    rewards = np.empty(N, dtype=float)

    for i, p in enumerate(batch_p):
        L1, L2, L3, L4, xO2, yO2 = p
        # Grashof / crank-rocker feasibility check
        if not is_crank_rocker_with_L2_crank(L1, L2, L3, L4):
            rewards[i] = -1e6
            continue

        coupler_pts, frac_valid, _ = coupler_path(p)

        if frac_valid < 0.99:
            rewards[i] = -1e6
            continue

        rewards[i] = trajectory_reward(coupler_pts, target_pts)

    return rewards
