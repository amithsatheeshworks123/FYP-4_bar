"""
PPO with feasibility repair, crank-rocker enforcement, and checkpointing.
"""

import os
import time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from v2.path_kinematics import is_crank_rocker_with_L2_crank


def repair(sample, bounds):
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    sample = np.clip(sample, low, high)
    lengths = sample[:4]
    if not is_crank_rocker_with_L2_crank(*lengths):
        arr = lengths.copy()
        s_idx = int(arr.argmin())
        l_idx = int(arr.argmax())
        # Put shortest at index 1 (L2 = crank).
        # Track where the maximum element moves if it was at index 1.
        if s_idx != 1:
            arr[[1, s_idx]] = arr[[s_idx, 1]]
            if l_idx == 1:   # max was at index 1 — it moved to s_idx
                l_idx = s_idx
        # Put longest at index 0 (L1 = ground).
        if l_idx != 0:
            arr[[0, l_idx]] = arr[[l_idx, 0]]
        sample[:4] = np.clip(arr, low[:4], high[:4])
    return sample


class Policy(nn.Module):
    def __init__(self, bounds, init_mean=None, init_log_std=None):
        super().__init__()
        lows = torch.tensor([b[0] for b in bounds])
        highs = torch.tensor([b[1] for b in bounds])
        self.register_buffer("lows", lows)
        self.register_buffer("highs", highs)
        base_mean = (lows + highs) / 2 if init_mean is None else torch.tensor(init_mean, dtype=torch.float32)
        base_log_std = torch.full_like(base_mean, -0.5) if init_log_std is None else torch.full_like(base_mean, float(init_log_std))
        self.mean = nn.Parameter(base_mean)
        self.log_std = nn.Parameter(base_log_std)

    def forward(self, batch_size):
        std = torch.exp(self.log_std)
        eps = torch.randn(batch_size, len(self.mean), device=self.mean.device)
        samples = self.mean + eps * std
        samples = torch.clamp(samples, self.lows, self.highs)
        log_probs = -0.5 * (((samples - self.mean) / (std + 1e-8)) ** 2 + 2 * self.log_std + np.log(2 * np.pi))
        return samples, log_probs.sum(dim=1)


def load_checkpoint(checkpoint_path, bounds, device, init_mean=None, init_log_std=None):
    policy = Policy(bounds, init_mean=init_mean, init_log_std=init_log_std).to(device)
    value = nn.Linear(len(bounds), 1).to(device)
    best_params = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Explicitly allow full checkpoint objects saved locally (PyTorch 2.6 made weights_only=True default).
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt.get("policy", {}))
        value.load_state_dict(ckpt.get("value", {}))
        best_params = ckpt.get("best_params", None)
    return policy, value, best_params


def save_checkpoint(checkpoint_path, policy, value, best_params):
    if not checkpoint_path:
        return
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({"policy": policy.state_dict(), "value": value.state_dict(), "best_params": best_params}, checkpoint_path)


def _default_baseline(bounds):
    """Chebyshev-inspired baseline: L1 (ground) longest, L2 (crank) shortest.

    Old ratios [4,2,5,5] made L3=L4 > L1, violating the L1-longest requirement
    and producing borderline Grashof.  Ratios [6,2,4,4.5] give:
      normalised ≈ [1.455, 0.485, 0.970, 1.091]
    so L1 > L4 > L3 > L2 with comfortable Grashof margin (s+l < p+q).
    """
    lens_ratios = np.array([6.0, 2.0, 4.0, 4.5])
    lens_ratios /= np.mean(lens_ratios)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    mids = 0.5 * (lows + highs)
    mid_len = np.mean(mids[:4])
    lengths = np.clip(mid_len * lens_ratios, lows[:4], highs[:4])
    x0 = mids[4]
    y0 = mids[5]
    return np.concatenate([lengths, [x0, y0]])


def _pool_eval(args):
    fn, arr = args
    return fn(np.array(arr, dtype=float))


def ppo_train(
    objective_fn,
    bounds,
    steps=5000,
    batch_size=128,
    lr=5e-4,
    seed=None,
    log_samples=False,
    checkpoint_path=None,
    num_workers=None,
    parallel_objective=True,
    baseline=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # Build baseline if none given
    if baseline is None:
        baseline = _default_baseline(bounds)
    # Initialize policy/value (checkpoint will override if present)
    policy, value, best_params = load_checkpoint(
        checkpoint_path,
        bounds,
        device,
        init_mean=baseline,
        init_log_std=-1.0,
    )
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=lr)
    # Seed best_r from checkpoint if available
    best_r = -1e9
    if best_params is not None:
        try:
            best_r = float(objective_fn(best_params))
        except Exception:
            best_r = -1e9
    else:
        best_params = None
        # ensure baseline is repaired as starting best guess
        best_params = None
    hist = []
    samples_log = [] if log_samples else None
    clip_eps = 0.2

    # multiprocessing pool for objective evaluation
    pool = None
    if parallel_objective:
        if num_workers is None:
            cpu_count = os.cpu_count() or 1
            num_workers = max(1, cpu_count - 1)
        try:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(num_workers, initializer=np.random.seed, initargs=(seed or 0,))
        except Exception as e:
            print(f"[PPO] Parallel init failed, falling back to serial eval: {e}")
            pool = None
            parallel_objective = False

    for step in range(steps):
        t0 = time.time()
        samples, log_probs = policy(batch_size)
        samples_np = samples.detach().cpu().numpy()
        repaired = np.stack([repair(s, bounds) for s in samples_np])
        t_sample = time.time()

        # objective evaluation (parallel if available)
        rewards_list = None
        if parallel_objective and pool is not None:
            try:
                rewards_list = pool.map(
                    _pool_eval, [(objective_fn, r.tolist()) for r in repaired]
                )
            except Exception as e:
                print(f"[PPO] Parallel eval failed, falling back to serial: {e}")
                rewards_list = None
                parallel_objective = False
        if rewards_list is None:
            rewards_list = []
            for s in repaired:
                lengths = s[:4]
                if np.any(lengths <= 0) or not is_crank_rocker_with_L2_crank(*lengths):
                    rewards_list.append(-1e6)
                else:
                    rewards_list.append(objective_fn(s))
        t_obj = time.time()

        repaired_t = torch.tensor(repaired, dtype=torch.float32, device=device)
        std = torch.exp(policy.log_std)
        log_probs_new = -0.5 * (((repaired_t - policy.mean) / (std + 1e-8)) ** 2 + 2 * policy.log_std + np.log(2 * np.pi))
        log_probs_new = log_probs_new.sum(dim=1)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        values = value(repaired_t).squeeze(1)
        advantages = rewards - values.detach()
        ratios = torch.exp(log_probs_new - log_probs)
        clipped = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(ratios * advantages, clipped).mean()
        value_loss = torch.nn.functional.mse_loss(values, rewards)
        # Entropy bonus: maximise Gaussian entropy = keep log_std from collapsing.
        # Entropy ∝ sum(log_std); subtracting from loss = maximising entropy.
        entropy_bonus = policy.log_std.mean()
        loss = policy_loss + 0.5 * value_loss - 1e-2 * entropy_bonus
        opt.zero_grad()
        loss.backward()
        opt.step()
        t_upd = time.time()

        hist.append(rewards.mean().item())
        if log_samples:
            for s, r in zip(repaired, rewards.cpu().numpy()):
                samples_log.append((step, float(r), s.copy()))
        max_idx = rewards.argmax().item()
        if rewards[max_idx] > best_r:
            best_r = rewards[max_idx].item()
            best_params = repaired[max_idx].copy()
        print(
            f"[PPO step {step+1}/{steps}] "
            f"sample={(t_sample - t0):.3f}s obj={(t_obj - t_sample):.3f}s upd={(t_upd - t_obj):.3f}s "
            f"mean_reward={rewards.mean().item():.3f}"
        )
    save_checkpoint(checkpoint_path, policy, value, best_params)
    if pool is not None:
        pool.close()
        pool.join()
    return best_r, best_params, hist, samples_log
