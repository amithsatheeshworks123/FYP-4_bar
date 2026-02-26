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

from path_kinematics import path_reward, THETA2, solve_rocker_angle
from optimizer import optimize
from cem import cem

app = Flask(__name__)

BOUNDS = [(0.05, 0.4)] * 4 + [(-0.1, 0.2), (-0.1, 0.2)]
DEFAULT_PARAMS = [0.1, 0.07, 0.12, 0.09, 0.0, 0.0]
PPO_CKPT = os.path.join(os.path.dirname(__file__), "ppo_checkpoint.pt")


def project(params, lock_mask=None, locked_vals=None):
    """Clip params to bounds, apply locks, and enforce Grashof crank-rocker."""
    arr = np.array(params, dtype=float)
    low = np.array([b[0] for b in BOUNDS])
    high = np.array([b[1] for b in BOUNDS])
    if lock_mask and locked_vals:
        for i, locked in enumerate(lock_mask):
            if locked:
                arr[i] = float(locked_vals[i])
    arr = np.clip(arr, low, high)
    Ls = arr[:4].copy()
    s_idx = int(Ls.argmin())
    l_idx = int(Ls.argmax())
    if s_idx != 1:
        Ls[[1, s_idx]] = Ls[[s_idx, 1]]
    if l_idx != 0:
        Ls[[0, l_idx]] = Ls[[l_idx, 0]]
    arr[:4] = Ls
    return arr


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics", methods=["POST"])
def compute_metrics():
    """Compute coupler path, rocker angles, and reward metrics for given params."""
    data = request.json or {}
    raw_params = data.get("params", DEFAULT_PARAMS)
    lock_mask = data.get("lock_mask", [False] * 6)
    target_c = float(data.get("target_c", 0.0))
    target_line = (0.0, target_c)

    params = project(raw_params)
    reward, fitted_line, pts, mse, max_dev = path_reward(
        params, target_line=target_line, enforce_grashof="crank_rocker"
    )

    rocker_angles = []
    for t2 in THETA2:
        th4 = solve_rocker_angle(t2, params)
        rocker_angles.append(None if th4 is None else float(th4 * 180.0 / np.pi))

    rom = None
    valid_th4 = [v for v in rocker_angles if v is not None]
    if valid_th4:
        rom = float(max(valid_th4) - min(valid_th4))

    coupler_pts = pts.tolist() if len(pts) > 0 else []

    return jsonify({
        "reward": float(reward),
        "mse": float(mse) if mse != float("inf") and mse == mse else None,
        "max_dev": float(max_dev) if max_dev != float("inf") and max_dev == max_dev else None,
        "coupler_pts": coupler_pts,
        "rocker_angles": rocker_angles,
        "crank_angles": THETA2.tolist(),
        "rom": rom,
        "params": params.tolist(),
    })


def _make_batch_obj(target_line):
    def obj(x):
        px = project(x)
        if px.ndim == 1:
            r, _, _, mse, max_dev = path_reward(px, target_line=target_line, enforce_grashof="crank_rocker")
            return np.array([float(r)])
        r_list = []
        for row in px:
            r, _, _, _, _ = path_reward(row, target_line=target_line, enforce_grashof="crank_rocker")
            r_list.append(float(r))
        return np.array(r_list)
    return obj


def _make_scalar_obj(target_line):
    def obj(x):
        px = project(x)
        if px.ndim == 1:
            r, _, _, _, _ = path_reward(px, target_line=target_line, enforce_grashof="crank_rocker")
            return float(r)
        r_list = []
        for row in px:
            r, _, _, _, _ = path_reward(row, target_line=target_line, enforce_grashof="crank_rocker")
            r_list.append(float(r))
        return np.array(r_list)
    return obj


@app.route("/api/optimize/cma", methods=["POST"])
def run_cma():
    """Run CMA-ES optimizer and return best parameters."""
    data = request.json or {}
    params = project(data.get("params", DEFAULT_PARAMS))
    target_c = float(data.get("target_c", 0.0))
    target_line = (0.0, target_c)
    seed_val = _parse_seed(data.get("seed"))

    obj = _make_batch_obj(target_line)
    best_r, best_p, hist, _ = optimize(
        obj, params, BOUNDS, iterations=60, population=120, restarts=3, seed=seed_val
    )
    best_p = project(best_p)
    _, _, pts, mse, max_dev = path_reward(best_p, target_line=target_line, enforce_grashof="crank_rocker")

    return jsonify({
        "best_params": best_p.tolist(),
        "reward": float(best_r),
        "mse": float(mse),
        "max_dev": float(max_dev),
        "history": [float(h) for h in hist],
    })


@app.route("/api/optimize/cem", methods=["POST"])
def run_cem():
    """Run Cross-Entropy Method optimizer and return best parameters."""
    data = request.json or {}
    params = project(data.get("params", DEFAULT_PARAMS))
    target_c = float(data.get("target_c", 0.0))
    target_line = (0.0, target_c)
    seed_val = _parse_seed(data.get("seed"))

    obj = _make_scalar_obj(target_line)
    best_r, best_p, hist = cem(
        obj, params, BOUNDS, iterations=80, pop=200, elite_frac=0.2, sigma=0.1, seed=seed_val
    )
    best_p = project(best_p)
    _, _, pts, mse, max_dev = path_reward(best_p, target_line=target_line, enforce_grashof="crank_rocker")

    return jsonify({
        "best_params": best_p.tolist(),
        "reward": float(best_r),
        "mse": float(mse),
        "max_dev": float(max_dev),
        "history": [float(h) for h in hist],
    })


@app.route("/api/optimize/ppo", methods=["POST"])
def run_ppo():
    """Run PPO reinforcement learning optimizer and return best parameters."""
    data = request.json or {}
    params = project(data.get("params", DEFAULT_PARAMS))
    target_c = float(data.get("target_c", 0.0))
    target_line = (0.0, target_c)
    seed_val = _parse_seed(data.get("seed"))

    try:
        from ppo import ppo_train

        def obj(x):
            r, _, _, _, _ = path_reward(project(x), target_line=target_line, enforce_grashof="crank_rocker")
            return float(r)

        best_r, best_p, hist, _ = ppo_train(
            obj, BOUNDS, steps=5000, batch_size=128, lr=5e-4,
            seed=seed_val, log_samples=False, checkpoint_path=PPO_CKPT,
        )
        best_p = project(best_p)
        _, _, pts, mse, max_dev = path_reward(best_p, target_line=target_line, enforce_grashof="crank_rocker")

        return jsonify({
            "best_params": best_p.tolist(),
            "reward": float(best_r),
            "mse": float(mse),
            "max_dev": float(max_dev),
            "history": [float(h) for h in hist],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _parse_seed(raw):
    if raw is None:
        return None
    try:
        s = str(raw).strip().lower()
        if s in ("", "none", "random"):
            return None
        return int(float(s))
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    print("Starting Four-Bar Linkage Synthesizer at http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
