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


def batch_path_reward(batch_p, target_line=None):
    """Fully-vectorised batch version of path_reward.

    Parameters
    ----------
    batch_p    : array-like, shape (N, 6)  — design vectors [L1..L4, xO2, yO2]
    target_line: (a, b) tuple or None.  None → fit best line per sample.

    Returns
    -------
    rewards : np.ndarray, shape (N,)   — same semantics as path_reward()[0]
    """
    batch_p = np.asarray(batch_p, dtype=float)
    N = batch_p.shape[0]
    rewards = np.full(N, -1e6, dtype=float)

    L1v = batch_p[:, 0]; L2v = batch_p[:, 1]
    L3v = batch_p[:, 2]; L4v = batch_p[:, 3]

    # ── 1. Basic positivity ──────────────────────────────────────────────────
    basic_ok = np.all(batch_p[:, :4] > 0, axis=1)

    # ── 2. Grashof crank-rocker: L2 must be shortest (idx 1), overall Grashof ─
    # Mirrors is_crank_rocker_with_L2_crank(): argmin==1 AND s+l<=others_sum.
    # Does NOT require L1 to be longest — project() handles that externally.
    arr4  = batch_p[:, :4]
    s_idx = arr4.argmin(axis=1)
    s     = arr4.min(axis=1)
    l     = arr4.max(axis=1)
    grashof_ok = (s_idx == 1) & (s + l <= arr4.sum(axis=1) - s - l + 1e-9)

    vi = np.where(basic_ok & grashof_ok)[0]
    if vi.size == 0:
        return rewards

    # ── 3. Vectorised kinematics  (M, 360) ───────────────────────────────────
    vp   = batch_p[vi]                   # (M, 6)
    L1  = vp[:, 0:1]; L2  = vp[:, 1:2]
    L3  = vp[:, 2:3]; L4  = vp[:, 3:4]
    xO2 = vp[:, 4:5]; yO2 = vp[:, 5:6]

    th2 = THETA2[np.newaxis, :]          # (1, 360)
    Bx  = xO2 + L2 * np.cos(th2)        # (M, 360)
    By  = yO2 + L2 * np.sin(th2)
    dBx = Bx - L1                        # O4x = L1, O4y = 0
    dBy = By
    R   = np.sqrt(dBx**2 + dBy**2)

    min_r  = np.abs(L3 - L4)            # (M, 1)
    max_r  = L3 + L4
    vstep  = (max_r > 0) & (R >= min_r - 1e-9) & (R <= max_r + 1e-9)  # (M, 360)

    # Only keep samples with ≥99 % valid steps (full crank rotation)
    all_ok = vstep.mean(axis=1) >= 0.99  # (M,)
    if not all_ok.any():
        return rewards

    # Rocker angle for valid steps
    R_s    = np.where(vstep, R, 1.0)
    cosPhi = np.clip((L4**2 + R_s**2 - L3**2) / (2.0 * L4 * R_s), -1.0, 1.0)
    phi    = np.arccos(cosPhi)
    th4    = np.arctan2(dBy, dBx) - phi  # (M, 360)

    Cx    = L1 + L4 * np.cos(th4)        # (M, 360)
    Cy    =      L4 * np.sin(th4)
    mid_x = 0.5 * (Bx + Cx)
    mid_y = 0.5 * (By + Cy)

    # ── 4. Transmission angle penalty over T_SAMPLES (60 pts, matches _transmission_penalty) ─
    th2_t  = T_SAMPLES[np.newaxis, :]           # (1, 60)
    Bx_t   = xO2 + L2 * np.cos(th2_t)          # (M, 60)
    By_t   = yO2 + L2 * np.sin(th2_t)
    dBx_t  = Bx_t - L1;  dBy_t = By_t
    R_t    = np.sqrt(dBx_t**2 + dBy_t**2)
    vs_t   = (max_r > 0) & (R_t >= min_r - 1e-9) & (R_t <= max_r + 1e-9)
    R_st   = np.where(vs_t, R_t, 1.0)
    cP_t   = np.clip((L4**2 + R_st**2 - L3**2) / (2.0*L4*R_st), -1.0, 1.0)
    ph_t   = np.arccos(cP_t)
    th4_t  = np.arctan2(dBy_t, dBx_t) - ph_t
    Cx_t   = L1 + L4 * np.cos(th4_t)
    Cy_t   = L4 * np.sin(th4_t)
    vc_xt  = Bx_t - Cx_t;  vc_yt = By_t - Cy_t
    vr_xt  = L1   - Cx_t;  vr_yt = -Cy_t
    nc_t   = np.sqrt(vc_xt**2 + vc_yt**2)
    nr_t   = np.sqrt(vr_xt**2 + vr_yt**2)
    gd_t   = vs_t & (nc_t > 1e-9) & (nr_t > 1e-9)
    cm_t   = np.where(gd_t, (vc_xt*vr_xt + vc_yt*vr_yt)/(nc_t*nr_t+1e-12), 0.0)
    mu_t   = np.degrees(np.arccos(np.clip(cm_t, -1.0, 1.0)))
    pen_t  = np.where(~vs_t, 10.0,
             np.where(~gd_t, 10.0,
             np.maximum(0.0, 40.0-mu_t)**2 + np.maximum(0.0, mu_t-140.0)**2))
    trans_pen = pen_t.mean(axis=1)              # (M,)

    # Only compute path statistics for fully-valid samples (avoids all-NaN warnings)
    ok_sub = np.where(all_ok)[0]
    if ok_sub.size == 0:
        return rewards

    mx_ok = np.where(vstep[ok_sub], mid_x[ok_sub], np.nan)   # (K, 360)
    my_ok = np.where(vstep[ok_sub], mid_y[ok_sub], np.nan)

    if target_line is not None:
        a_arr = np.full(ok_sub.size, float(target_line[0]))
        b_arr = np.full(ok_sub.size, float(target_line[1]))
    else:
        n_v  = vstep[ok_sub].sum(axis=1).astype(float)
        sx   = np.nansum(np.where(vstep[ok_sub], mid_x[ok_sub],            0.0), axis=1)
        sy   = np.nansum(np.where(vstep[ok_sub], mid_y[ok_sub],            0.0), axis=1)
        sx2  = np.nansum(np.where(vstep[ok_sub], mid_x[ok_sub]**2,         0.0), axis=1)
        sxy  = np.nansum(np.where(vstep[ok_sub], mid_x[ok_sub]*mid_y[ok_sub], 0.0), axis=1)
        denom = n_v * sx2 - sx**2 + 1e-12
        a_arr = (n_v * sxy - sx * sy) / denom
        b_arr = (sy - a_arr * sx) / (n_v + 1e-12)

    y_fit = a_arr[:, np.newaxis] * mx_ok + b_arr[:, np.newaxis]
    err   = y_fit - my_ok
    mse   = np.nanmean(err**2,     axis=1)
    maxd  = np.nanmax(np.abs(err), axis=1)

    dx         = np.nanmax(mx_ok, axis=1) - np.nanmin(mx_ok, axis=1)
    dy         = np.nanmax(my_ok, axis=1) - np.nanmin(my_ok, axis=1)
    aspect     = dy / (dx + EPS)
    straight   = np.nanstd(my_ok, axis=1) / (dx + EPS)
    stroke_pen = np.maximum(0.0, DX_MIN - dx)

    cost  = (W_STRAIGHT * straight
             + W_ASPECT  * aspect**2
             + W_STROKE  * stroke_pen**2
             + W_MAXDEV  * maxd)
    r_vec = -cost - 0.05 * trans_pen[ok_sub]

    rewards[vi[ok_sub]] = r_vec
    return rewards


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
