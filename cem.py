"""
Cross-Entropy Method for path-generation task.
"""

import numpy as np


def cem(objective_fn, init_mean, bounds, iterations=80, pop=200, elite_frac=0.2, sigma=0.1, seed=None):
    """Cross-Entropy Method.

    objective_fn may be either:
      - a scalar function  f(x: 1-D array) → float, or
      - a batch  function  f(X: 2-D array (N,D)) → 1-D array of floats.
    A batch-capable objective (e.g. batch_path_reward) is used when available
    for a significant speedup; falls back to per-sample otherwise.
    """
    rng = np.random.default_rng(seed)
    low  = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    span = high - low
    mean = np.array(init_mean, dtype=float)
    std  = np.maximum(span * sigma, 1e-3)
    elite_count = max(1, int(pop * elite_frac))
    best_r = -np.inf
    best_p = mean.copy()
    hist   = []

    # Probe whether objective_fn accepts a batch (2-D input)
    _probe = np.tile(mean, (2, 1))
    try:
        _r = objective_fn(_probe)
        batch_mode = isinstance(_r, (np.ndarray, list)) and len(np.asarray(_r)) == 2
    except Exception:
        batch_mode = False

    for _ in range(iterations):
        samples = rng.normal(scale=std, size=(pop, len(mean))) + mean
        samples = np.clip(samples, low, high)

        if batch_mode:
            rewards = np.asarray(objective_fn(samples), dtype=float)
        else:
            rewards = np.array([float(objective_fn(s)) for s in samples])

        idx           = rewards.argsort()[::-1][:elite_count]
        elites        = samples[idx]
        elite_rewards = rewards[idx]
        mean = elites.mean(axis=0)
        std  = elites.std(axis=0) + 1e-3
        if elite_rewards[0] > best_r:
            best_r = float(elite_rewards[0])
            best_p = elites[0].copy()
        hist.append(float(rewards.mean()))

    return best_r, best_p, hist
