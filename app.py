"""
Flask web server for the Four-Bar Linkage Synthesizer.

Run:
    cd /home/user/FYP-4_bar
    python app.py

Then open http://localhost:5000 in your browser.

Install dependencies first:
    pip install flask numpy
"""

import sys
import os
import numpy as np
from flask import Flask, render_template, request, jsonify

# Add this directory to sys.path so we can import the kinematics modules directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_kinematics import (
    path_reward, batch_path_reward, THETA2, solve_rocker_angle
)
from optimizer import optimize
from cem import cem
from target_trajectory import (
    get_trajectory_types, generate_trajectory,
    trajectory_path_reward, batch_trajectory_reward,
)

app = Flask(__name__)

BOUNDS          = [(0.05, 0.4)] * 4 + [(-0.1, 0.2), (-0.1, 0.2)]
DEFAULT_PARAMS  = [0.1, 0.07, 0.12, 0.09, 0.0, 0.0]
PPO_CKPT        = os.path.join(os.path.dirname(__file__), "ppo_checkpoint.pt")


# ── Helpers ───────────────────────────────────────────────────────────────────

def project(params, lock_mask=None, locked_vals=None):
    """Clip to bounds, apply locks, enforce Grashof crank-rocker."""
    arr  = np.array(params, dtype=float)
    low  = np.array([b[0] for b in BOUNDS])
    high = np.array([b[1] for b in BOUNDS])
    if lock_mask and locked_vals:
        for i, locked in enumerate(lock_mask):
            if locked:
                arr[i] = float(locked_vals[i])
    arr  = np.clip(arr, low, high)
    Ls   = arr[:4].copy()
    s_idx = int(Ls.argmin())
    l_idx = int(Ls.argmax())
    if s_idx != 1:
        Ls[[1, s_idx]] = Ls[[s_idx, 1]]
    if l_idx != 0:
        Ls[[0, l_idx]] = Ls[[l_idx, 0]]
    arr[:4] = Ls
    return arr


def project_batch(batch, lock_mask=None, locked_vals=None):
    """Project a (N,6) batch."""
    return np.stack([project(row, lock_mask, locked_vals) for row in batch])


def _parse_seed(raw):
    if raw is None:
        return None
    try:
        s = str(raw).strip().lower()
        return None if s in ("", "none", "random") else int(float(s))
    except (ValueError, TypeError):
        return None


def _parse_traj(data):
    """Extract trajectory type and params from request data.

    Falls back to 'straight_line' with target_c for backward compatibility.
    """
    traj_type   = data.get("traj_type", "straight_line")
    traj_params = data.get("traj_params") or {}
    # Backward compat: old clients send target_c instead of traj_params
    if not traj_params and "target_c" in data:
        traj_params = {"c": float(data.get("target_c", 0.0))}
    return traj_type, traj_params


# ── Batch objective wrappers ──────────────────────────────────────────────────

def make_batch_obj(target_line, lock_mask=None, locked_vals=None):
    """Return a function that accepts (N,6) and returns (N,) rewards."""
    def obj(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X = project_batch(X, lock_mask, locked_vals)
        return batch_path_reward(X, target_line=target_line)
    return obj


def make_scalar_obj(target_line, lock_mask=None, locked_vals=None):
    """Scalar wrapper for optimisers that call per-sample (CEM fallback)."""
    def obj(x):
        return float(batch_path_reward(
            project(x, lock_mask, locked_vals)[np.newaxis, :],
            target_line=target_line
        )[0])
    return obj


def _make_traj_batch_obj(traj_type, traj_params, lock_mask=None, locked_vals=None):
    """Create a batch objective for any trajectory type."""
    if traj_type == "straight_line":
        c  = float(traj_params.get("c", 0.0))
        tl = (0.0, c)
        return make_batch_obj(tl, lock_mask, locked_vals)
    else:
        target_pts = generate_trajectory(traj_type, traj_params)

        def obj(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X[np.newaxis, :]
            X = project_batch(X, lock_mask, locked_vals)
            return batch_trajectory_reward(X, target_pts)
        return obj


def _eval_traj(params, traj_type, traj_params):
    """Evaluate reward for any trajectory type.

    Returns (reward, coupler_pts, target_pts_list, mse, max_dev).
    """
    target_pts = generate_trajectory(traj_type, traj_params)
    if traj_type == "straight_line":
        c  = float(traj_params.get("c", 0.0))
        tl = (0.0, c)
        reward, _, pts, mse, max_dev = path_reward(
            params, target_line=tl, enforce_grashof="crank_rocker"
        )
    else:
        reward, pts, mse, max_dev = trajectory_path_reward(
            params, target_pts, enforce_grashof="crank_rocker"
        )
    tp_list = target_pts.tolist() if len(target_pts) > 0 else []
    return reward, pts, tp_list, mse, max_dev


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/trajectories", methods=["GET"])
def list_trajectories():
    """Return available trajectory types and their parameter schemas."""
    return jsonify(get_trajectory_types())


@app.route("/api/trajectory/preview", methods=["POST"])
def preview_trajectory():
    """Generate and return target trajectory points for canvas preview."""
    data        = request.json or {}
    traj_type   = data.get("traj_type", "straight_line")
    traj_params = data.get("traj_params") or {}
    pts         = generate_trajectory(traj_type, traj_params)
    return jsonify({"points": pts.tolist() if len(pts) > 0 else []})


@app.route("/api/metrics", methods=["POST"])
def compute_metrics():
    """Compute coupler path, rocker angles, and reward metrics for given params."""
    data      = request.json or {}
    params    = project(data.get("params", DEFAULT_PARAMS))
    traj_type, traj_params = _parse_traj(data)

    reward, pts, target_pts, mse, max_dev = _eval_traj(params, traj_type, traj_params)

    rocker_angles = []
    for t2 in THETA2:
        th4 = solve_rocker_angle(t2, params)
        rocker_angles.append(None if th4 is None else float(th4 * 180.0 / np.pi))

    valid_th4 = [v for v in rocker_angles if v is not None]
    rom = float(max(valid_th4) - min(valid_th4)) if valid_th4 else None

    return jsonify({
        "reward":        float(reward),
        "mse":           None if not np.isfinite(mse)     else float(mse),
        "max_dev":       None if not np.isfinite(max_dev) else float(max_dev),
        "coupler_pts":   pts.tolist() if len(pts) > 0 else [],
        "target_pts":    target_pts,
        "rocker_angles": rocker_angles,
        "crank_angles":  THETA2.tolist(),
        "rom":           rom,
        "params":        params.tolist(),
    })


@app.route("/api/optimize/cma", methods=["POST"])
def run_cma():
    """Run CMA-ES optimiser (batch objective)."""
    data       = request.json or {}
    params     = project(data.get("params", DEFAULT_PARAMS))
    traj_type, traj_params = _parse_traj(data)
    seed_val   = _parse_seed(data.get("seed"))

    obj = _make_traj_batch_obj(traj_type, traj_params)
    best_r, best_p, hist, _ = optimize(
        obj, params, BOUNDS, iterations=60, population=120, restarts=3, seed=seed_val
    )
    best_p = project(best_p)
    reward, _, target_pts, mse, max_dev = _eval_traj(best_p, traj_type, traj_params)

    return jsonify({
        "best_params": best_p.tolist(),
        "reward":      float(reward),
        "mse":         float(mse)     if np.isfinite(mse)     else None,
        "max_dev":     float(max_dev) if np.isfinite(max_dev) else None,
        "history":     [float(h) for h in hist],
    })


@app.route("/api/optimize/cem", methods=["POST"])
def run_cem():
    """Run CEM optimiser (batch objective)."""
    data       = request.json or {}
    params     = project(data.get("params", DEFAULT_PARAMS))
    traj_type, traj_params = _parse_traj(data)
    seed_val   = _parse_seed(data.get("seed"))

    obj = _make_traj_batch_obj(traj_type, traj_params)
    best_r, best_p, hist = cem(
        obj, params, BOUNDS, iterations=80, pop=200, elite_frac=0.2, sigma=0.1, seed=seed_val
    )
    best_p = project(best_p)
    reward, _, target_pts, mse, max_dev = _eval_traj(best_p, traj_type, traj_params)

    return jsonify({
        "best_params": best_p.tolist(),
        "reward":      float(reward),
        "mse":         float(mse)     if np.isfinite(mse)     else None,
        "max_dev":     float(max_dev) if np.isfinite(max_dev) else None,
        "history":     [float(h) for h in hist],
    })


@app.route("/api/optimize/ppo", methods=["POST"])
def run_ppo():
    """
    Run PPO optimiser.

    Default: cold-start from the same starting point as CMA-ES and CEM,
    ensuring a fair comparison between all three methods.

    Optional warm-start: pass {"warm_start": true} in the request body to
    first run CMA-ES and seed PPO from its result (clearly labelled in response).
    """
    data            = request.json or {}
    params          = project(data.get("params", DEFAULT_PARAMS))
    traj_type, traj_params = _parse_traj(data)
    seed_val        = _parse_seed(data.get("seed"))
    steps           = int(data.get("steps", 300))
    warm_start_mode = bool(data.get("warm_start", False))

    try:
        from ppo import ppo_train

        # Build trajectory-specific batch objective for PPO
        traj_batch_obj = _make_traj_batch_obj(traj_type, traj_params) \
                         if traj_type != "straight_line" else None

        if traj_type == "straight_line":
            c  = float(traj_params.get("c", 0.0))
            tl = (0.0, c)
        else:
            tl = None  # batch_obj handles the reward

        if warm_start_mode:
            cma_obj = _make_traj_batch_obj(traj_type, traj_params)
            cma_r, cma_p, _, _ = optimize(
                cma_obj, params, BOUNDS,
                iterations=40, population=100, restarts=2, seed=seed_val
            )
            ppo_baseline = project(cma_p)
            mode_label   = "PPO + CMA-ES warm-start"
            print(f"[PPO warm-start] CMA-ES best reward: {cma_r:.4f}")
        else:
            ppo_baseline = params
            mode_label   = "PPO cold-start"
            print("[PPO cold-start] Using default params as baseline (fair comparison)")

        def scalar_obj(x):
            if traj_type == "straight_line":
                return float(batch_path_reward(
                    project(x)[np.newaxis, :], target_line=tl
                )[0])
            else:
                tp = generate_trajectory(traj_type, traj_params)
                return float(batch_trajectory_reward(project(x)[np.newaxis, :], tp)[0])

        best_r, best_p, hist, _ = ppo_train(
            scalar_obj,
            BOUNDS,
            steps=steps,
            batch_size=256,
            lr=3e-4,
            seed=seed_val,
            log_samples=False,
            checkpoint_path=PPO_CKPT,
            baseline=ppo_baseline,
            early_stop_reward=-0.3,
            target_line=tl,
            batch_obj=traj_batch_obj,
        )
        best_p = project(best_p)
        reward, _, target_pts, mse, max_dev = _eval_traj(best_p, traj_type, traj_params)

        return jsonify({
            "best_params": best_p.tolist(),
            "reward":      float(reward),
            "mse":         float(mse)     if np.isfinite(mse)     else None,
            "max_dev":     float(max_dev) if np.isfinite(max_dev) else None,
            "history":     [float(h) for h in hist],
            "mode":        mode_label,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/optimize/ppo_extended", methods=["POST"])
def run_ppo_extended():
    """
    Run PPO with extended training for prolonged optimisation.

    Default: cold-start (same starting point as CMA-ES and CEM) for a fair
    comparison. Pass {"warm_start": true} to optionally pre-run CMA-ES first.
    """
    data            = request.json or {}
    params          = project(data.get("params", DEFAULT_PARAMS))
    traj_type, traj_params = _parse_traj(data)
    seed_val        = _parse_seed(data.get("seed"))
    steps           = int(data.get("steps", 5000))
    warm_start_mode = bool(data.get("warm_start", False))

    try:
        from ppo import ppo_train

        traj_batch_obj = _make_traj_batch_obj(traj_type, traj_params) \
                         if traj_type != "straight_line" else None

        if traj_type == "straight_line":
            c  = float(traj_params.get("c", 0.0))
            tl = (0.0, c)
        else:
            tl = None

        if warm_start_mode:
            cma_obj = _make_traj_batch_obj(traj_type, traj_params)
            cma_r, cma_p, _, _ = optimize(
                cma_obj, params, BOUNDS,
                iterations=80, population=150, restarts=3, seed=seed_val
            )
            ppo_baseline = project(cma_p)
            mode_label   = "PPO-Extended + CMA-ES warm-start"
            print(f"[PPO-Extended warm-start] CMA-ES best reward: {cma_r:.4f}")
        else:
            ppo_baseline = params
            mode_label   = "PPO-Extended cold-start"
            print("[PPO-Extended cold-start] Using default params as baseline (fair comparison)")

        def scalar_obj(x):
            if traj_type == "straight_line":
                return float(batch_path_reward(
                    project(x)[np.newaxis, :], target_line=tl
                )[0])
            else:
                tp = generate_trajectory(traj_type, traj_params)
                return float(batch_trajectory_reward(project(x)[np.newaxis, :], tp)[0])

        best_r, best_p, hist, _ = ppo_train(
            scalar_obj,
            BOUNDS,
            steps=steps,
            batch_size=512,
            lr=1e-4,
            seed=seed_val,
            log_samples=False,
            checkpoint_path=PPO_CKPT,
            baseline=ppo_baseline,
            early_stop_reward=-0.1,
            target_line=tl,
            batch_obj=traj_batch_obj,
        )
        best_p = project(best_p)
        reward, _, target_pts, mse, max_dev = _eval_traj(best_p, traj_type, traj_params)

        return jsonify({
            "best_params": best_p.tolist(),
            "reward":      float(reward),
            "mse":         float(mse)     if np.isfinite(mse)     else None,
            "max_dev":     float(max_dev) if np.isfinite(max_dev) else None,
            "history":     [float(h) for h in hist],
            "mode":        mode_label,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/optimize/ppo_sequential", methods=["POST"])
def run_ppo_sequential():
    """
    Run the sequential MDP PPO optimiser (FourBarEnv, 6-step episode).

    The agent selects one design parameter per step (L1→L2→L3→L4→xO2→yO2),
    receiving intermediate feasibility rewards at steps 1 (L2<L1?) and 3
    (Grashof?) plus the full path_reward at step 5.

    Same JSON request/response format as the other optimizer routes.
    """
    data       = request.json or {}
    traj_type, traj_params = _parse_traj(data)
    seed_val   = _parse_seed(data.get("seed"))
    steps      = int(data.get("steps", 300))

    try:
        from ppo_sequential import ppo_sequential_train

        if traj_type == "straight_line":
            c  = float(traj_params.get("c", 0.0))
            tl = (0.0, c)
            traj_batch_obj = None
        else:
            tl             = None
            traj_batch_obj = _make_traj_batch_obj(traj_type, traj_params)

        best_r, best_p, hist, _ = ppo_sequential_train(
            BOUNDS,
            steps       = steps,
            batch_size  = 256,
            lr          = 3e-4,
            seed        = seed_val,
            target_line = tl,
            log_samples = False,
            batch_obj   = traj_batch_obj,
        )
        best_p = project(best_p)
        reward, _, target_pts, mse, max_dev = _eval_traj(best_p, traj_type, traj_params)

        return jsonify({
            "best_params": best_p.tolist(),
            "reward":      float(reward),
            "mse":         float(mse)     if np.isfinite(mse)     else None,
            "max_dev":     float(max_dev) if np.isfinite(max_dev) else None,
            "history":     [float(h) for h in hist],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Four-Bar Linkage Synthesizer at http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
