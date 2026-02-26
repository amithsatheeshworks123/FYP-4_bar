"""
Kinematics and reward for straight-line path generation of a planar four-bar.
Design vector p = [L1, L2, L3, L4, xO2, yO2].
Ground: O1=(0,0), O4=(L1,0); input pivot O2=(xO2,yO2); input crank angle theta2.
Coupler point: midpoint of link B-C.
"""

import numpy as np

THETA2 = np.linspace(0, 2 * np.pi, 360)
T_SAMPLES = np.linspace(0, 2 * np.pi, 60)  # coarse grid for transmission angle

# Straight-line reward tuning parameters
DX_MIN = 0.3  # minimum horizontal stroke (m); solutions with dx < DX_MIN are heavily penalized
W_STRAIGHT = 1.5  # weight on straightness (std(y)/dx)
W_ASPECT = 1.25  # weight on aspect ratio (dy/dx)^2 to suppress vertical swing
W_STROKE = 5.0  # weight on horizontal stroke penalty when dx < DX_MIN
W_MAXDEV = 1.0  # weight on absolute max deviation from the fitted line
EPS = 1e-9


def grashof_crank_rocker(lengths):
    """Return True if lengths satisfy crank-rocker: shortest is crank (L2), ground (L1) longest, Grashof."""
    L1, L2, L3, L4 = lengths
    arr = np.array([L1, L2, L3, L4])
    s_idx = arr.argmin()
    l_idx = arr.argmax()
    s = arr[s_idx]
    l = arr[l_idx]
    others = np.sort(arr)[1:3]
    return s_idx == 1 and l_idx == 0 and (s + l <= others.sum() + 1e-9)


def is_grashof(L1, L2, L3, L4, eps=1e-9):
    arr = np.array([L1, L2, L3, L4])
    s = arr.min()
    l = arr.max()
    others = np.sort(arr)[1:3]
    return s + l <= others.sum() + eps


def is_crank_rocker_with_L2_crank(L1, L2, L3, L4, eps=1e-9):
    arr = np.array([L1, L2, L3, L4])
    if not is_grashof(L1, L2, L3, L4, eps=eps):
        return False
    if arr.argmin() != 1:  # L2 must be shortest (crank)
        return False
    return True


def closure(theta2, p):
    L1, L2, L3, L4, xO2, yO2 = p
    O1 = np.array([0.0, 0.0])
    O2 = np.array([xO2, yO2])
    O4 = np.array([L1, 0.0])
    B = O2 + np.array([L2 * np.cos(theta2), L2 * np.sin(theta2)])
    R = np.linalg.norm(B - O4)
    min_r = abs(L3 - L4)
    max_r = L3 + L4
    if max_r <= 0 or R < min_r or R > max_r:
        return None
    cos_phi = (L4**2 + R**2 - L3**2) / (2 * L4 * R)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    psi = np.arctan2(B[1] - O4[1], B[0] - O4[0])
    theta4 = psi - phi
    C = O4 + np.array([L4 * np.cos(theta4), L4 * np.sin(theta4)])
    coupler_mid = B + 0.5 * (C - B)
    return B, C, O4, coupler_mid


def coupler_path(p):
    """Vectorised coupler midpoint path over all THETA2 angles."""
    L1, L2, L3, L4, xO2, yO2 = p
    O4x = float(L1)
    Bx = xO2 + L2 * np.cos(THETA2)
    By = yO2 + L2 * np.sin(THETA2)
    dBx = Bx - O4x
    dBy = By  # O4y = 0
    R = np.sqrt(dBx ** 2 + dBy ** 2)
    min_r = abs(L3 - L4)
    max_r = L3 + L4
    valid = (max_r > 0) & (R >= min_r - 1e-9) & (R <= max_r + 1e-9)
    R_safe = np.where(valid, R, 1.0)
    cos_phi = np.clip((L4 ** 2 + R_safe ** 2 - L3 ** 2) / (2.0 * L4 * R_safe), -1.0, 1.0)
    phi = np.arccos(cos_phi)
    theta4 = np.arctan2(dBy, dBx) - phi
    Cx = O4x + L4 * np.cos(theta4)
    Cy = L4 * np.sin(theta4)
    mid_x = 0.5 * (Bx + Cx)
    mid_y = 0.5 * (By + Cy)
    valid_count = int(valid.sum())
    total = len(THETA2)
    if valid_count == 0:
        return np.empty((0, 2)), 0.0, False
    pts_valid = np.column_stack([mid_x[valid], mid_y[valid]])
    return pts_valid, valid_count / total, valid_count == total


def solve_rocker_angle(theta2, p):
    """Return rocker angle theta4 for param vector p = [L1..L4, xO2, yO2]."""
    L1, L2, L3, L4, xO2, yO2 = p
    O2 = np.array([xO2, yO2])
    O4 = np.array([L1, 0.0])
    B = O2 + np.array([L2 * np.cos(theta2), L2 * np.sin(theta2)])
    R = np.linalg.norm(B - O4)
    min_r = abs(L3 - L4)
    max_r = L3 + L4
    if max_r <= 0 or R < min_r or R > max_r:
        return None
    cos_phi = (L4**2 + R**2 - L3**2) / (2 * L4 * R)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    psi = np.arctan2(B[1] - O4[1], B[0] - O4[0])
    theta4 = psi - phi
    return theta4


def transmission_angle(theta2, p):
    """Angle between coupler (B->C) and rocker (C->D) at joint C (deg)."""
    res = closure(theta2, p)
    if res is None:
        return None
    B, C, D, _ = res
    v_coup = B - C
    v_rock = D - C
    nc = np.linalg.norm(v_coup)
    nr = np.linalg.norm(v_rock)
    if nc < 1e-9 or nr < 1e-9:
        return None
    cos_mu = np.clip(np.dot(v_coup, v_rock) / (nc * nr), -1.0, 1.0)
    return np.degrees(np.arccos(cos_mu))


def _transmission_penalty(p):
    """Vectorised transmission angle penalty over T_SAMPLES.

    Replaces the 60-iteration Python loop in path_reward.
    Returns the mean penalty (same scale as original loop / len(T_SAMPLES)).
    """
    L1, L2, L3, L4, xO2, yO2 = p
    O4x = float(L1)
    Bx = xO2 + L2 * np.cos(T_SAMPLES)
    By = yO2 + L2 * np.sin(T_SAMPLES)
    dBx = Bx - O4x
    dBy = By  # O4y = 0
    R = np.sqrt(dBx ** 2 + dBy ** 2)
    min_r = abs(L3 - L4)
    max_r = L3 + L4
    valid = (max_r > 0) & (R >= min_r - 1e-9) & (R <= max_r + 1e-9)
    R_safe = np.where(valid, R, 1.0)
    cos_phi = np.clip((L4 ** 2 + R_safe ** 2 - L3 ** 2) / (2.0 * L4 * R_safe), -1.0, 1.0)
    phi = np.arccos(cos_phi)
    theta4 = np.arctan2(dBy, dBx) - phi
    Cx = O4x + L4 * np.cos(theta4)
    Cy = L4 * np.sin(theta4)
    # transmission angle: angle at C between coupler (C→B) and rocker (C→O4)
    vc_x = Bx - Cx;  vc_y = By - Cy   # coupler direction
    vr_x = O4x - Cx; vr_y = -Cy        # rocker direction (O4y=0)
    nc = np.sqrt(vc_x ** 2 + vc_y ** 2)
    nr = np.sqrt(vr_x ** 2 + vr_y ** 2)
    good = valid & (nc > 1e-9) & (nr > 1e-9)
    cos_mu = np.where(good, (vc_x * vr_x + vc_y * vr_y) / (nc * nr + 1e-12), 0.0)
    cos_mu = np.clip(cos_mu, -1.0, 1.0)
    mu = np.degrees(np.arccos(cos_mu))
    # match original: 10.0 for invalid/degenerate; quadratic for out-of-range
    pen = np.where(
        ~valid, 10.0,
        np.where(~good, 10.0,
                 np.maximum(0.0, 40.0 - mu) ** 2 + np.maximum(0.0, mu - 140.0) ** 2)
    )
    return float(pen.mean())


def path_reward(p, target_line=None, enforce_grashof=None):
    """Reward: negative path error to straight line over full rotation."""
    p = np.array(p, dtype=float)
    default_line = target_line if target_line is not None else (0.0, 0.0)
    if np.any(p[:4] <= 0):
        return -1e6, default_line, np.empty((0, 2)), np.inf, np.inf
    if enforce_grashof == "crank_rocker":
        if not is_crank_rocker_with_L2_crank(*p[:4]):
            return -1e6, default_line, np.empty((0, 2)), np.inf, np.inf
    # compute path (single sweep only)
    pts, valid_frac, all_valid = coupler_path(p)
    if (not all_valid) or valid_frac < 0.99 or len(pts) < len(THETA2) * 0.99:
        return -1e6, default_line, np.empty((0, 2)), np.inf, np.inf
    if target_line is None:
        # Fit best line y = ax + b
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        a, b = target_line
    y_pred = a * pts[:, 0] + b
    err = y_pred - pts[:, 1]
    mse = np.mean(err**2)
    max_dev = np.max(np.abs(err))
    # ----- Straight-line focused reward with stroke enforcement -----
    x = pts[:, 0]
    y = pts[:, 1]
    dx = float(np.max(x) - np.min(x))
    dy = float(np.max(y) - np.min(y))
    aspect = dy / (dx + EPS)  # vertical-to-horizontal ratio
    straight = float(np.std(y) / (dx + EPS))  # scale-aware straightness (penalizes tiny motion)
    stroke_pen = max(0.0, DX_MIN - dx)
    # Include absolute max deviation to further discourage large vertical excursions
    cost = (
        W_STRAIGHT * straight
        + W_ASPECT * (aspect**2)
        + W_STROKE * (stroke_pen**2)
        + W_MAXDEV * max_dev
    )
    reward = -cost
    # Transmission angle penalty (vectorised over T_SAMPLES)
    pen = _transmission_penalty(p)
    reward -= 0.05 * pen
    return float(reward), (a, b), pts, float(mse), float(max_dev)
