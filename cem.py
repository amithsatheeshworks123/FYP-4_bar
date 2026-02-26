"""
Cross-Entropy Method for path-generation task.
"""

import numpy as np


def cem(objective_fn, init_mean, bounds, iterations=80, pop=200, elite_frac=0.2, sigma=0.1, seed=None):
    rng = np.random.default_rng(seed)
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    span = high - low
    mean = np.array(init_mean, dtype=float)
    std = np.maximum(span * sigma, 1e-3)
    elite_count = max(1, int(pop * elite_frac))
    best_r = -np.inf
    best_p = mean.copy()
    hist = []
    for _ in range(iterations):
        samples = rng.normal(scale=std, size=(pop, len(mean))) + mean
        samples = np.clip(samples, low, high)
        rewards = np.array([objective_fn(s) for s in samples])
        idx = rewards.argsort()[::-1][:elite_count]
        elites = samples[idx]
        elite_rewards = rewards[idx]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3
        if elite_rewards[0] > best_r:
            best_r = elite_rewards[0]
            best_p = elites[0].copy()
        hist.append(rewards.mean())
    return best_r, best_p, hist
