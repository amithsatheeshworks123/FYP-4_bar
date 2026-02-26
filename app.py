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


def _target_line(target_c):
    return (0.0, float(target_c))


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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics", methods=["POST"])
def compute_metrics():
    """Compute coupler path, rocker angles, and reward metrics for given params."""
    data      = request.json or {}
    params    = project(data.get("params", DEFAULT_PARAMS))
    target_c  = float(data.get("target_c", 0.0))
    tl        = _target_line(target_c)

    reward, fitted_line, pts, mse, max_dev = path_reward(
        params, target_line=tl, enforce_grashof="crank_rocker"
    )

    rocker_angles = []
    for t2 in THETA2:
        th4 = solve_rocker_angle(t2, params)
        rocker_angles.append(None if th4 is None else float(th4 * 180.0 / np.pi))

    valid_th4 = [v for v in rocker_angles if v is not None]
    rom = float(max(valid_th4) - min(valid_th4)) if valid_th4 else None

    return jsonify({
        "reward":       float(reward),
        "mse":          None if not np.isfinite(mse)     else float(mse),
        "max_dev":      None if not np.isfinite(max_dev) else float(max_dev),
        "coupler_pts":  pts.tolist() if len(pts) > 0 else [],
        "rocker_angles": rocker_angles,
        "crank_angles": THETA2.tolist(),
        "rom":          rom,
        "params":       params.tolist(),
    })


@app.route("/api/optimize/cma", methods=["POST"])
def run_cma():
    """Run CMA-ES optimiser (batch objective)."""
    data     = request.json or {}
    params   = project(data.get("params", DEFAULT_PARAMS))
    tl       = _target_line(data.get("target_c", 0.0))
    seed_val = _parse_seed(data.get("seed"))

    obj = make_batch_obj(tl)
    best_r, best_p, hist, _ = optimize(
        obj, params, BOUNDS, iterations=60, population=120, restarts=3, seed=seed_val
    )
    best_p = project(best_p)
    _, _, _, mse, max_dev = path_reward(best_p, target_line=tl, enforce_grashof="crank_rocker")

    return jsonify({
        "best_params": best_p.tolist(),
        "reward":      float(best_r),
        "mse":         float(mse),
        "max_dev":     float(max_dev),
        "history":     [float(h) for h in hist],
    })


@app.route("/api/optimize/cem", methods=["POST"])
def run_cem():
    """Run CEM optimiser (batch objective)."""
    data     = request.json or {}
    params   = project(data.get("params", DEFAULT_PARAMS))
    tl       = _target_line(data.get("target_c", 0.0))
    seed_val = _parse_seed(data.get("seed"))

    # Pass batch obj — cem.py auto-detects batch vs scalar
    obj = make_batch_obj(tl)
    best_r, best_p, hist = cem(
        obj, params, BOUNDS, iterations=80, pop=200, elite_frac=0.2, sigma=0.1, seed=seed_val
    )
    best_p = project(best_p)
    _, _, _, mse, max_dev = path_reward(best_p, target_line=tl, enforce_grashof="crank_rocker")

    return jsonify({
        "best_params": best_p.tolist(),
        "reward":      float(best_r),
        "mse":         float(mse),
        "max_dev":     float(max_dev),
        "history":     [float(h) for h in hist],
    })


@app.route("/api/optimize/ppo", methods=["POST"])
def run_ppo():
    """
    Run PPO optimiser.
    Warm-starts from CMA-ES result for much faster convergence:
      1. Run CMA-ES (fast, ~5 s) to find a good basin
      2. Seed the PPO policy mean from the CMA-ES solution
      3. Run PPO for fewer steps with tighter initial std
    """
    data     = request.json or {}
    params   = project(data.get("params", DEFAULT_PARAMS))
    tl       = _target_line(data.get("target_c", 0.0))
    seed_val = _parse_seed(data.get("seed"))
    steps    = int(data.get("steps", 300))           # reduced default

    try:
        from ppo import ppo_train

        # ── Step 1: quick CMA-ES warm-start ───────────────────────────────────
        cma_obj   = make_batch_obj(tl)
        cma_r, cma_p, _, _ = optimize(
            cma_obj, params, BOUNDS,
            iterations=40, population=100, restarts=2, seed=seed_val
        )
        warm_start = project(cma_p)
        print(f"[PPO warm-start] CMA-ES best reward: {cma_r:.4f}")

        # ── Step 2: PPO from warm start ────────────────────────────────────────
        # Scalar objective is only used for checkpoint seeding; batch_path_reward
        # is called directly inside ppo_train via target_line kwarg.
        def scalar_obj(x):
            return float(batch_path_reward(
                project(x)[np.newaxis, :], target_line=tl
            )[0])

        best_r, best_p, hist, _ = ppo_train(
            scalar_obj,
            BOUNDS,
            steps=steps,
            batch_size=256,
            lr=3e-4,
            seed=seed_val,
            log_samples=False,
            checkpoint_path=PPO_CKPT,
            baseline=warm_start,          # seed policy mean from CMA-ES
            early_stop_reward=-0.3,       # stop early if reward is good enough
            target_line=tl,               # batch_path_reward inside ppo_train
        )
        best_p = project(best_p)
        _, _, _, mse, max_dev = path_reward(best_p, target_line=tl, enforce_grashof="crank_rocker")

        return jsonify({
            "best_params": best_p.tolist(),
            "reward":      float(best_r),
            "mse":         float(mse),
            "max_dev":     float(max_dev),
            "history":     [float(h) for h in hist],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/optimize/ppo_extended", methods=["POST"])
def run_ppo_extended():
    """
    Run PPO with extended training for prolonged optimisation.
    Trains for significantly more steps than the standard PPO run, allowing
    the policy to converge to a higher-quality solution:
      1. Run a thorough CMA-ES warm-start to find a strong initial basin
      2. Seed the PPO policy from the CMA-ES result
      3. Run PPO for many steps with a lower learning rate and larger batch
         so the policy refines gradually without overshooting
    """
    data     = request.json or {}
    params   = project(data.get("params", DEFAULT_PARAMS))
    tl       = _target_line(data.get("target_c", 0.0))
    seed_val = _parse_seed(data.get("seed"))
    steps    = int(data.get("steps", 1000))          # 3× longer than standard

    try:
        from ppo import ppo_train

        # ── Step 1: thorough CMA-ES warm-start ────────────────────────────────
        cma_obj = make_batch_obj(tl)
        cma_r, cma_p, _, _ = optimize(
            cma_obj, params, BOUNDS,
            iterations=80, population=150, restarts=3, seed=seed_val
        )
        warm_start = project(cma_p)
        print(f"[PPO-Extended warm-start] CMA-ES best reward: {cma_r:.4f}")

        # ── Step 2: extended PPO from warm start ───────────────────────────────
        def scalar_obj(x):
            return float(batch_path_reward(
                project(x)[np.newaxis, :], target_line=tl
            )[0])

        best_r, best_p, hist, _ = ppo_train(
            scalar_obj,
            BOUNDS,
            steps=steps,
            batch_size=512,           # larger batch for smoother gradients
            lr=1e-4,                  # lower lr for careful fine-tuning
            seed=seed_val,
            log_samples=False,
            checkpoint_path=PPO_CKPT,
            baseline=warm_start,
            early_stop_reward=-0.1,   # only stop early if result is excellent
            target_line=tl,
        )
        best_p = project(best_p)
        _, _, _, mse, max_dev = path_reward(best_p, target_line=tl, enforce_grashof="crank_rocker")

        return jsonify({
            "best_params": best_p.tolist(),
            "reward":      float(best_r),
            "mse":         float(mse),
            "max_dev":     float(max_dev),
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
