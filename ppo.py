"""
PPO with:
  - Vectorised batch objective (no multiprocessing needed — NumPy is already fast)
  - Deeper policy (MLP mean head) + deeper value network
  - Cosine-annealed entropy coefficient (explore → exploit)
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
    """Small MLP value function V(x) for advantage estimation."""
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
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=lr
    )

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

    for step in range(steps):
        t0 = time.time()

        # ── Sample + repair ───────────────────────────────────────────────────
        samples, log_probs = policy(batch_size)
        samples_np = samples.detach().cpu().numpy()
        repaired   = batch_repair(samples_np, bounds)

        # ── Vectorised reward evaluation ──────────────────────────────────────
        rewards_np = batch_path_reward(repaired, target_line=target_line)
        t_eval = time.time()

        repaired_t = torch.tensor(repaired,    dtype=torch.float32, device=device)
        rewards    = torch.tensor(rewards_np,  dtype=torch.float32, device=device)

        # Re-compute log-probs under current policy for repaired samples
        mean = policy.get_mean()
        std  = torch.exp(policy.log_std)
        lp_new = -0.5 * (((repaired_t - mean) / (std + 1e-8))**2
                         + 2 * policy.log_std
                         + math.log(2 * math.pi))
        lp_new = lp_new.sum(dim=1)

        # ── PPO update ────────────────────────────────────────────────────────
        values     = value(repaired_t)
        advantages = rewards - values.detach()
        ratios     = torch.exp(lp_new - log_probs)
        clipped    = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(ratios * advantages, clipped).mean()
        value_loss  = nn.functional.mse_loss(values, rewards)

        # Cosine-annealed entropy: high at start (explore), low at end (exploit)
        entropy_coef = 0.05 * 0.5 * (1.0 + math.cos(math.pi * step / steps))
        entropy_bonus = policy.log_std.mean()
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy_bonus

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()
        t_upd = time.time()

        mean_r = float(rewards.mean())
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
            f"mean={mean_r:.4f}  best={best_r:.4f}"
        )

        # Early stopping
        if early_stop_reward is not None and best_r >= early_stop_reward:
            print(f"[PPO] Early stop at step {step+1}: best_r={best_r:.4f}")
            break

    save_checkpoint(checkpoint_path, policy, value, best_params)
    return best_r, best_params, hist, samples_log
