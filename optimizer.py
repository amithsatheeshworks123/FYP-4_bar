"""
Constraint-aware CMA-ES style optimizer (v2).
Resamples valid elites only; per-dimension sigma based on bounds.
"""

import numpy as np
from typing import Callable, Tuple


def optimize(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    init_params: np.ndarray,
    bounds,
    iterations: int = 80,
    population: int = 120,
    elite_frac: float = 0.2,
    restarts: int = 3,
    seed: int = None,
) -> Tuple[float, np.ndarray, list, list]:
    rng = np.random.default_rng(seed)
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    span = high - low
    best_reward = -np.inf
    best_params = np.array(init_params, dtype=float)
    history = []
    best_history = []
    elite_count = max(1, int(population * elite_frac))

    def run_once(start_mean):
        nonlocal best_reward, best_params
        mean = start_mean.copy()
        sigma = np.maximum(span * 0.1, 0.02)
        for _ in range(iterations):
            samples = rng.normal(scale=sigma, size=(population, len(mean))) + mean
            samples = np.clip(samples, low, high)
            rewards = np.array(objective_fn(samples))
            valid_mask = rewards > -1e8
            if not valid_mask.any():
                sigma *= 0.7
                continue
            samples = samples[valid_mask]
            rewards = rewards[valid_mask]
            idx = rewards.argsort()[::-1][:elite_count]
            elites = samples[idx]
            elite_rewards = rewards[idx]
            mean = elites.mean(axis=0)
            sigma = elites.std(axis=0) + 1e-3
            if elite_rewards[0] > best_reward:
                best_reward = float(elite_rewards[0])
                best_params = elites[0].copy()
            history.append(float(rewards.mean()))
            best_history.append(best_reward)

    for r in range(max(1, restarts)):
        if r == 0:
            start = np.array(init_params, dtype=float)
        else:
            start = low + rng.random(len(init_params)) * span
        run_once(start)

    return best_reward, best_params, history, best_history
