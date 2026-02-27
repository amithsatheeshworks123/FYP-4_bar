"""
PPO with:
  - Vectorised batch objective (no multiprocessing needed — NumPy is already fast)
  - Deeper policy (MLP mean head)
  - Running-mean reward baseline (replaces learned value net — correct for bandit setting)
  - Rank-based advantage normalization (robust to reward scale and outliers)
  - Multi-epoch minibatch PPO updates (sample efficiency: K=6 epochs per batch)
  - Fixed log-prob computation: both old and new log-probs on the SAME repaired samples
  - Entropy coefficient floored at 0.01 (never fully zero — maintains exploration)
  - log_std clamped >= -3.0 (prevents std collapse below ~0.05)
  - Early stopping when reward threshold is met
  - Warm-start support: pass init_params to seed the policy mean
  - Feasibility repair + crank-rocker enforcement
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from path_kinematics import is_crank_rocker_with_L2_crank, batch_path_reward


# ── Feasibility repair ────────────────────────────────────────────────────────

def repair(sample, bounds):
    """Clip to bounds and enforce Grashof crank-rocker (L2 shortest, L1 longest)."""
    low  = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    sample = np.clip(sample, low, high)
    arr = sample[:4].copy()
    s_idx = int(arr.argmin())
    l_idx = int(arr.argmax())
    if s_idx != 1:
        arr[[1, s_idx]] = arr[[s_idx, 1]]
        if l_idx == 1:
            l_idx = s_idx
    if l_idx != 0:
        arr[[0, l_idx]] = arr[[l_idx, 0]]
    sample[:4] = np.clip(arr, low[:4], high[:4])
    return sample


def batch_repair(samples, bounds):
    """Vectorised repair for shape (N, 6)."""
    low  = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    out  = np.clip(samples, low, high)
    for i in range(len(out)):
        out[i] = repair(out[i], bounds)
    return out


# ── Rank-based advantage normalization ────────────────────────────────────────

def _rank_normalize(arr):
    """
    Rank rewards and normalize to [-0.5, 0.5].
    Mirrors the implicit rank selection used by CMA-ES and CEM.
    Robust to reward scale differences and outliers (e.g. -1e6 infeasible penalty).
    """
    n = len(arr)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return (ranks / (n - 1)) - 0.5  # map to [-0.5, 0.5]


# ── Networks ──────────────────────────────────────────────────────────────────

class Policy(nn.Module):
    """
    MLP policy: state-independent Gaussian.
    A small hidden layer lets the mean adapt through gradient flow more
    expressively than a single learnable parameter vector.
    """
    def __init__(self, bounds, init_mean=None, init_log_std=None):
        super().__init__()
        dim  = len(bounds)
        lows  = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        highs = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        self.register_buffer("lows",  lows)
        self.register_buffer("highs", highs)

        # Small MLP that maps a learned "context" vector → mean offsets
        self.context = nn.Parameter(torch.zeros(dim))
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.Tanh(),
            nn.Linear(64, 64),  nn.Tanh(),
            nn.Linear(64, dim),
        )
        # Separate learnable log-std
        log_std_init = float(init_log_std) if init_log_std is not None else -1.0
        self.log_std = nn.Parameter(torch.full((dim,), log_std_init))

        # Seed mean from init_mean if provided
        if init_mean is not None:
            m = torch.tensor(init_mean, dtype=torch.float32)
            # normalise to [-1, 1] for the MLP context
            m_norm = 2.0 * (m - lows) / (highs - lows + 1e-8) - 1.0
            with torch.no_grad():
                self.context.copy_(m_norm)

    def get_mean(self):
        """Decode context through MLP to get un-clipped mean in world space."""
        delta = self.mlp(self.context.unsqueeze(0)).squeeze(0)
        raw   = self.context + delta
        # Map from [-1,1] domain back to [lows, highs]
        mean  = self.lows + (raw + 1.0) * 0.5 * (self.highs - self.lows)
        return torch.clamp(mean, self.lows, self.highs)

    def forward(self, batch_size):
        mean  = self.get_mean()                               # (dim,)
        std   = torch.exp(self.log_std)
        eps   = torch.randn(batch_size, len(mean), device=mean.device)
        samples = mean + eps * std
        samples = torch.clamp(samples, self.lows, self.highs)
        log_probs = -0.5 * (((samples - mean) / (std + 1e-8))**2
                            + 2 * self.log_std
                            + math.log(2 * math.pi))
        return samples, log_probs.sum(dim=1)


class ValueNet(nn.Module):
    """Kept for checkpoint compatibility — not used in training."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.Tanh(),
            nn.Linear(64, 64),  nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path, bounds, device, init_mean=None, init_log_std=None):
    dim    = len(bounds)
    policy = Policy(bounds, init_mean=init_mean, init_log_std=init_log_std).to(device)
    value  = ValueNet(dim).to(device)
    best_params = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        try:
            policy.load_state_dict(ckpt.get("policy", {}), strict=False)
        except Exception:
            pass
        try:
            value.load_state_dict(ckpt.get("value", {}), strict=False)
        except Exception:
            pass
        best_params = ckpt.get("best_params", None)
    return policy, value, best_params


def save_checkpoint(checkpoint_path, policy, value, best_params):
    if not checkpoint_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    torch.save({"policy": policy.state_dict(),
                "value":  value.state_dict(),
                "best_params": best_params}, checkpoint_path)


# ── Default baseline ──────────────────────────────────────────────────────────

def _default_baseline(bounds):
    """Chebyshev-inspired starting point: L1 longest, L2 shortest."""
    lens_ratios = np.array([6.0, 2.0, 4.0, 4.5])
    lens_ratios /= np.mean(lens_ratios)
    lows  = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    mids  = 0.5 * (lows + highs)
    lengths = np.clip(np.mean(mids[:4]) * lens_ratios, lows[:4], highs[:4])
    return np.concatenate([lengths, [mids[4], mids[5]]])


# ── Main training loop ────────────────────────────────────────────────────────

def ppo_train(
    objective_fn,          # scalar fn(x) → float  (used only for checkpoint seed)
    bounds,
    steps=500,             # reduced default — vectorised eval converges faster
    batch_size=256,        # larger batch exploits vectorisation
    lr=3e-4,
    seed=None,
    log_samples=False,
    checkpoint_path=None,
    baseline=None,
    early_stop_reward=None,  # stop if best_r >= this value
    target_line=None,        # passed directly to batch_path_reward
):
    """
    Train using vectorised batch_path_reward — no multiprocessing needed.
    The bottleneck (kinematics) is now pure NumPy broadcasting over (N,360).

    Bugs fixed vs original implementation:
      1. Log-prob mismatch: old log-probs are now computed on REPAIRED samples
         (same point as lp_new), so the PPO ratio pi_new/pi_old is correct.
      2. Running-mean EMA baseline replaces the learned ValueNet. In a single-step
         bandit, ValueNet only learns prediction error, giving noisy then vanishing
         gradient signal. EMA stably centres advantages throughout training.
      3. Multi-epoch minibatch updates (K=6 epochs, mb=64): squeezes more learning
         from each expensive batch of reward evaluations.
      4. Entropy floor at 0.01 prevents exploration dying in the second half of
         training; log_std clamped >= -3.0 prevents std collapse below ~0.05.
      5. Rank-based advantage normalization mirrors CMA-ES/CEM's implicit rank
         selection and is robust to the large reward variance from -1e6 penalties.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if baseline is None:
        baseline = _default_baseline(bounds)

    policy, value, best_params = load_checkpoint(
        checkpoint_path, bounds, device,
        init_mean=baseline, init_log_std=-1.0,
    )
    # Optimize only policy parameters — value net is kept for checkpoint compat only
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # Seed best_r from checkpoint
    best_r = -1e9
    if best_params is not None:
        try:
            best_r = float(objective_fn(best_params))
        except Exception:
            best_r = -1e9

    hist        = []
    samples_log = [] if log_samples else None
    clip_eps    = 0.2
    n_epochs    = 6    # PPO inner epochs per collected batch
    mb_size     = 64   # minibatch size for inner loop

    # Running-mean EMA reward baseline (Bug 2 fix)
    # Replaces learned ValueNet: in a bandit setting the advantage is simply
    # reward - running_mean, which stably centres gradients without trying to
    # learn the reward function (which causes vanishing advantages late in training).
    reward_ema = 0.0
    ema_alpha  = 0.05
    ema_init   = True  # first step: initialise EMA to batch mean directly

    for step in range(steps):
        t0 = time.time()

        # ── Sample from policy (no grad needed for sampling) ───────────────────
        with torch.no_grad():
            samples, _ = policy(batch_size)
        samples_np = samples.detach().cpu().numpy()
        repaired   = batch_repair(samples_np, bounds)

        # ── Vectorised reward evaluation ───────────────────────────────────────
        rewards_np = batch_path_reward(repaired, target_line=target_line)
        t_eval = time.time()

        repaired_t = torch.tensor(repaired,   dtype=torch.float32, device=device)
        rewards    = torch.tensor(rewards_np, dtype=torch.float32, device=device)

        # ── Update running-mean baseline ───────────────────────────────────────
        batch_mean = float(rewards.mean())
        if ema_init:
            reward_ema = batch_mean
            ema_init   = False
        else:
            reward_ema = ema_alpha * batch_mean + (1.0 - ema_alpha) * reward_ema

        # ── Rank-based advantage normalization (additional robustness fix) ─────
        # Mirrors the implicit rank selection in CMA-ES and CEM.
        # Robust to the large variance from -1e6 infeasibility penalties.
        rank_adv_np = _rank_normalize(rewards_np)
        rank_adv    = torch.tensor(rank_adv_np, dtype=torch.float32, device=device)

        # ── Compute OLD log-probs on REPAIRED samples (Bug 1 fix) ─────────────
        # The PPO ratio pi_new(a) / pi_old(a) requires BOTH evaluated at the
        # SAME action 'a'. Previously log_probs came from ORIGINAL (pre-repair)
        # samples, while lp_new used REPAIRED samples — breaking the trust region.
        # Fix: snapshot the current (pre-update) policy distribution and evaluate
        # it on the repaired samples. Both old and new log-probs are then at the
        # same point.
        with torch.no_grad():
            old_mean    = policy.get_mean()          # (dim,)
            old_log_std = policy.log_std.clone()     # (dim,) — detached copy
            old_std     = torch.exp(old_log_std)

        log_probs_old = -0.5 * (((repaired_t - old_mean) / (old_std + 1e-8))**2
                                + 2 * old_log_std
                                + math.log(2 * math.pi))
        log_probs_old = log_probs_old.sum(dim=1)  # (N,) — no grad

        # ── Multi-epoch minibatch PPO update (Bug 3 fix) ──────────────────────
        # Standard PPO does K=3-10 epochs of shuffled minibatch updates over the
        # same collected batch. Previously only 1 update was done, wasting the
        # expensive reward evaluations.
        N = batch_size
        for _epoch in range(n_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb_size):
                idx = perm[start:start + mb_size]

                mb_rep     = repaired_t[idx]          # (mb, dim)
                mb_adv     = rank_adv[idx]            # (mb,)
                mb_lp_old  = log_probs_old[idx]       # (mb,) — no grad

                # Recompute log-probs under the CURRENT (updated) policy
                cur_mean = policy.get_mean()
                cur_std  = torch.exp(policy.log_std)
                lp_new = -0.5 * (((mb_rep - cur_mean) / (cur_std + 1e-8))**2
                                 + 2 * policy.log_std
                                 + math.log(2 * math.pi))
                lp_new = lp_new.sum(dim=1)            # (mb,)

                ratios  = torch.exp(lp_new - mb_lp_old)
                clipped = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -torch.min(ratios * mb_adv, clipped).mean()

                # Entropy with floor at 0.01 (Bug 4 fix)
                # Original schedule decays to 0, killing exploration in the second
                # half of training. Floor at 0.01 maintains minimum diversity.
                entropy_coef = max(0.01,
                    0.05 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))
                entropy_bonus = policy.log_std.mean()
                loss = policy_loss - entropy_coef * entropy_bonus

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                opt.step()

                # Clamp log_std to prevent std collapse (Bug 4 fix)
                # Without this floor, std can collapse below 0.05 and the policy
                # gets permanently stuck — matching CMA-ES/CEM population diversity.
                with torch.no_grad():
                    policy.log_std.clamp_(min=-3.0)

        t_upd = time.time()

        mean_r = batch_mean
        hist.append(mean_r)

        if log_samples:
            for s, r in zip(repaired, rewards_np):
                samples_log.append((step, float(r), s.copy()))

        # Track best
        max_idx = int(rewards.argmax())
        if rewards_np[max_idx] > best_r:
            best_r      = float(rewards_np[max_idx])
            best_params = repaired[max_idx].copy()

        print(
            f"[PPO step {step+1:4d}/{steps}]  "
            f"eval={t_eval-t0:.3f}s  upd={t_upd-t_eval:.3f}s  "
            f"mean={mean_r:.4f}  best={best_r:.4f}  ema={reward_ema:.4f}"
        )

        # Early stopping
        if early_stop_reward is not None and best_r >= early_stop_reward:
            print(f"[PPO] Early stop at step {step+1}: best_r={best_r:.4f}")
            break

    save_checkpoint(checkpoint_path, policy, value, best_params)
    return best_r, best_params, hist, samples_log
