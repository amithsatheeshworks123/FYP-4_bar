"""
CMA-ES optimizer using the pycma package (full covariance matrix adaptation).
"""

import numpy as np
import cma
from typing import Callable, Tuple


def optimize(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    init_params: np.ndarray,
    bounds,
    iterations: int = 80,
    population: int = 120,
    elite_frac: float = 0.2,  # ignored — CMA-ES handles selection internally
    restarts: int = 3,
    seed: int = None,
) -> Tuple[float, np.ndarray, list, list]:
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    span = high - low
    sigma0 = 0.3 * float(np.mean(span))

    best_reward = -np.inf
    best_params = np.array(init_params, dtype=float)
    history = []
    best_history = []

    cma_opts = {
        'popsize': population,
        'bounds': [low.tolist(), high.tolist()],
        'maxiter': iterations,
        'tolfun': 1e-11,
        'tolx': 1e-11,
        'CMA_active': True,
        'verbose': -9,  # suppress all output
    }
    if seed is not None:
        cma_opts['seed'] = int(seed)

    rng = np.random.default_rng(seed)

    for r in range(max(1, restarts)):
        if r == 0:
            x0 = np.array(init_params, dtype=float)
        else:
            x0 = low + rng.random(len(init_params)) * span

        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, cma_opts)

        while not es.stop():
            candidates = es.ask()  # list of lists, len == population
            X = np.clip(np.array(candidates), low, high)
            rewards = np.asarray(objective_fn(X), dtype=float)
            # CMA-ES minimizes, our objective maximizes — negate
            fitnesses = (-rewards).tolist()
            es.tell(candidates, fitnesses)

            history.append(float(rewards.mean()))
            gen_best = float(rewards.max())
            if gen_best > best_reward:
                best_reward = gen_best
                best_params = X[int(np.argmax(rewards))].copy()
            best_history.append(best_reward)

    return best_reward, best_params, history, best_history
