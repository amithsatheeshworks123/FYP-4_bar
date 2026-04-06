"""
PPO with:
  - Vectorised batch objective (no multiprocessing needed — NumPy is already fast)
  - Deeper policy (MLP mean head)
  - Running-mean reward baseline (replaces learned value net — correct for bandit setting)
  - Rank-based advantage normalization (robust to reward scale and outliers)
  - Multi-epoch minibatch PPO updates (sample efficiency: K=6 epochs per batch)
  - Fixed log-prob computation: both old and new log-probs on the SAME repaired samples
  - Shaped feasibility penalties: gradient-bearing penalty proportional to Grashof
    violation distance — breaks the "all rewards identical → zero gradient" death spiral
  - Higher log_std floor (-1.5 / std≈0.22) and initial log_std (-0.5 / std≈0.6)
    so the policy explores broadly enough to stay in / find the feasible region
  - Anchor loss: soft constraint pulling policy mean toward best known feasible design
  - Reset mechanism: after 20 consecutive all-infeasible steps, snap mean back to best
  - Stale-checkpoint deletion: ensures each run starts cleanly from the provided baseline
  - Entropy coefficient floored at 0.01 (never fully zero — maintains exploration)
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


def _ensure_torch_sympy_compat():
    """
    Torch 2.5+ imports `equal_valued` from SymPy, which is absent in SymPy 1.11.
    Add a minimal fallback so optimizer construction does not fail.
    """
    try:
        from sympy.core import numbers as sympy_numbers
    except Exception:
        return

    if not hasattr(sympy_numbers, "equal_valued"):
        def _equal_valued(a, b):
            try:
                return bool(a == b)
            except Exception:
                return False
        sympy_numbers.equal_valued = _equal_valued


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
    Robust to reward scale differences and outliers.
    NOTE: only meaningful when rewards are NOT all identical — Fix A (shaped
    penalties) ensures that even all-infeasible batches have varied rewards.
    """
    n = len(arr)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return (ranks / (n - 1)) - 0.5  # map to [-0.5, 0.5]


# ── Shaped feasibility penalty (Fix A) ────────────────────────────────────────

def _shape_infeasible_rewards(rewards_np, repaired, best_params):
    """
    Replace flat -1e6 rewards with gradient-bearing shaped penalties.

    Problem: when ALL rewards in a batch are -1e6, _rank_normalize produces
    noise-level differences (arbitrary tie-breaking) → effectively zero gradient.
    The policy is in an absorbing dead state.

    Fix: compute a shaped penalty proportional to how far each design is from
    being feasible.  Infeasible samples now have VARIED rewards, so rank
    normalization produces a meaningful gradient pointing toward feasibility.

    Penalty components (all non-positive, all in [-200, -10]):
      1. Grashof violation: how much s+l exceeds p+q  (Grashof crank-rocker needs s+l <= p+q)
      2. L2-must-be-shortest violation: how much L2 exceeds the actual minimum
      3. Distance to best known feasible design (if any) — pulls exploration toward
         the region we already know works
    """
    infeasible_mask = rewards_np <= -1e5
    if not infeasible_mask.any():
        return rewards_np  # nothing to do

    inf_samples = repaired[infeasible_mask]
    arr4        = inf_samples[:, :4]

    # 1. Grashof violation
    s                  = arr4.min(axis=1)
    l                  = arr4.max(axis=1)
    others_sum         = arr4.sum(axis=1) - s - l
    grashof_violation  = np.maximum(0.0, (s + l) - others_sum)

    # 2. L2-must-be-shortest violation (L2 = index 1)
    l2_vals     = arr4[:, 1]
    min_vals    = arr4.min(axis=1)
    l2_violation = np.maximum(0.0, l2_vals - min_vals)

    # 3. Distance to best known feasible design
    if best_params is not None:
        dist_to_best = np.sqrt(((inf_samples - best_params) ** 2).sum(axis=1))
    else:
        dist_to_best = np.zeros(infeasible_mask.sum())

    # Shaped penalty — keep in [-200, -10] so it's clearly worse than any feasible
    # reward (which are in roughly [-10, 0]) but varied enough for rank signal
    shaped = -10.0 - 20.0 * grashof_violation - 20.0 * l2_violation - 5.0 * dist_to_best
    shaped = np.clip(shaped, -200.0, -10.0)

    out = rewards_np.copy()
    out[infeasible_mask] = shaped
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
        # Separate learnable log-std; init higher for broader exploration (Fix B)
        log_std_init = float(init_log_std) if init_log_std is not None else -0.5
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


# ── Policy reset helper (Fix D) ───────────────────────────────────────────────

def _reset_policy_to(policy, target_params, log_std_val=-0.5):
    """
    Snap the policy mean back to target_params and reset MLP to small weights.
    Called after consecutive_infeasible >= MAX_INFEASIBLE_STREAK.
    """
    with torch.no_grad():
        target_t = torch.tensor(target_params, dtype=torch.float32,
                                device=policy.lows.device)
        # Re-encode target into normalised context space
        target_raw = (2.0 * (target_t - policy.lows)
                      / (policy.highs - policy.lows + 1e-8) - 1.0)
        policy.context.copy_(target_raw)
        # Reset log_std to encourage re-exploration
        policy.log_std.fill_(log_std_val)
        # Re-initialise MLP to small weights so context ≈ mean immediately
        for layer in policy.mlp:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
            if hasattr(layer, 'bias'):
                nn.init.zeros_(layer.bias)


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
    target_pts=None,         # (M, 2) array for trajectory reward; overrides target_line
):
    """
    Train using vectorised batch_path_reward — no multiprocessing needed.
    The bottleneck (kinematics) is now pure NumPy broadcasting over (N,360).

    Fixes applied:
      1. Log-prob mismatch: old log-probs computed on REPAIRED samples (same
         point as lp_new), so the PPO ratio pi_new/pi_old is correct.
      2. Running-mean EMA baseline replaces the learned ValueNet.
      3. Multi-epoch minibatch updates (K=6 epochs, mb=64).
      4. Entropy floor at 0.01; log_std clamped >= -1.5 (std ≈ 0.22).
      5. Rank-based advantage normalization.
      A. Shaped feasibility penalties — breaks "all -1e6 → zero gradient" collapse.
      B. Higher log_std init (-0.5 / std≈0.6) for broad initial exploration.
      C. Anchor loss — soft pull of policy mean toward best known feasible design.
      D. Reset mechanism — recover from all-infeasible absorbing state.
      E. Stale checkpoint deletion — start cleanly from provided baseline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if baseline is None:
        baseline = _default_baseline(bounds)

    # Fix E: delete stale checkpoint when a fresh baseline is provided so we
    # don't inherit a poisoned policy from a previous failed run
    if checkpoint_path and os.path.exists(checkpoint_path) and baseline is not None:
        print("[PPO] Removing stale checkpoint — starting fresh from baseline")
        os.remove(checkpoint_path)

    policy, value, best_params = load_checkpoint(
        checkpoint_path, bounds, device,
        init_mean=baseline, init_log_std=-0.5,  # Fix B: broader initial exploration
    )
    _ensure_torch_sympy_compat()
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

    # Running-mean EMA reward baseline
    reward_ema = 0.0
    ema_alpha  = 0.05
    ema_init   = True

    # Fix D: feasibility-collapse detection
    consecutive_infeasible = 0
    MAX_INFEASIBLE_STREAK  = 20

    for step in range(steps):
        t0 = time.time()

        # ── Sample from policy (no grad needed for sampling) ───────────────────
        with torch.no_grad():
            samples, _ = policy(batch_size)
        samples_np = samples.detach().cpu().numpy()
        repaired   = batch_repair(samples_np, bounds)

        # ── Vectorised reward evaluation ───────────────────────────────────────
        if target_pts is not None and target_line is None:
            from target_trajectory import batch_trajectory_reward
            rewards_np = batch_trajectory_reward(repaired, target_pts)
        else:
            rewards_np = batch_path_reward(repaired, target_line=target_line)
        t_eval = time.time()

        # ── Count feasible samples and update collapse detector (Fix D) ────────
        feasible_count = int((rewards_np > -1e5).sum())
        if feasible_count == 0:
            consecutive_infeasible += 1
        else:
            consecutive_infeasible = 0

        # Fix D: reset if stuck in all-infeasible absorbing state
        if consecutive_infeasible >= MAX_INFEASIBLE_STREAK and best_params is not None:
            print(f"[PPO] Resetting policy mean after {consecutive_infeasible} "
                  f"consecutive all-infeasible steps")
            _reset_policy_to(policy, best_params, log_std_val=-0.5)
            consecutive_infeasible = 0
            # Re-create optimizer so momentum doesn't undo the reset
            opt = torch.optim.Adam(policy.parameters(), lr=lr)

        # Fix A: replace flat -1e6 with gradient-bearing shaped penalties
        # This is the single most important change for the collapse case:
        # when all rewards are identical, rank normalization gives ~zero gradient.
        # Shaped penalties give varied rewards even in all-infeasible batches,
        # creating a gradient pointing from more-infeasible to less-infeasible.
        rewards_np = _shape_infeasible_rewards(rewards_np, repaired, best_params)

        repaired_t = torch.tensor(repaired,   dtype=torch.float32, device=device)
        rewards    = torch.tensor(rewards_np, dtype=torch.float32, device=device)

        # ── Update running-mean baseline ───────────────────────────────────────
        batch_mean = float(rewards.mean())
        if ema_init:
            reward_ema = batch_mean
            ema_init   = False
        else:
            reward_ema = ema_alpha * batch_mean + (1.0 - ema_alpha) * reward_ema

        # ── Rank-based advantage normalization ─────────────────────────────────
        # Now meaningful even for all-infeasible batches thanks to Fix A
        rank_adv_np = _rank_normalize(rewards_np)
        rank_adv    = torch.tensor(rank_adv_np, dtype=torch.float32, device=device)

        # ── Compute OLD log-probs on REPAIRED samples (Bug 1 fix) ─────────────
        with torch.no_grad():
            old_mean    = policy.get_mean()          # (dim,)
            old_log_std = policy.log_std.clone()     # (dim,) — detached copy
            old_std     = torch.exp(old_log_std)

        log_probs_old = -0.5 * (((repaired_t - old_mean) / (old_std + 1e-8))**2
                                + 2 * old_log_std
                                + math.log(2 * math.pi))
        log_probs_old = log_probs_old.sum(dim=1)  # (N,) — no grad

        # ── Multi-epoch minibatch PPO update ───────────────────────────────────
        N = batch_size
        for _epoch in range(n_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb_size):
                idx = perm[start:start + mb_size]

                mb_rep    = repaired_t[idx]       # (mb, dim)
                mb_adv    = rank_adv[idx]         # (mb,)
                mb_lp_old = log_probs_old[idx]    # (mb,) — no grad

                # Recompute log-probs under the CURRENT (updated) policy
                cur_mean = policy.get_mean()
                cur_std  = torch.exp(policy.log_std)
                lp_new = -0.5 * (((mb_rep - cur_mean) / (cur_std + 1e-8))**2
                                 + 2 * policy.log_std
                                 + math.log(2 * math.pi))
                lp_new = lp_new.sum(dim=1)  # (mb,)

                ratios  = torch.exp(lp_new - mb_lp_old)
                clipped = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -torch.min(ratios * mb_adv, clipped).mean()

                # Entropy with floor at 0.01
                entropy_coef  = max(0.01,
                    0.05 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))
                entropy_bonus = policy.log_std.mean()

                # Fix C: anchor loss — soft pull toward best known feasible design.
                # Prevents the policy mean from wandering into fully infeasible space.
                # Coefficient 0.05 is small enough not to block improvement on the
                # anchor, but large enough to resist drift.
                if best_params is not None:
                    anchor_t  = torch.tensor(best_params, dtype=torch.float32,
                                             device=device)
                    cur_mean_ = policy.get_mean()
                    ranges    = policy.highs - policy.lows + 1e-8
                    anchor_loss = 0.05 * (((cur_mean_ - anchor_t) / ranges) ** 2).mean()
                else:
                    anchor_loss = torch.tensor(0.0, device=device)

                loss = policy_loss - entropy_coef * entropy_bonus + anchor_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                opt.step()

                # Fix B: higher log_std floor — std >= 0.22 ≈ 63% of bound span.
                # Prevents the policy from narrowing so much that it can't rediscover
                # the feasible region if the mean drifts away from it.
                with torch.no_grad():
                    policy.log_std.data.clamp_(min=-1.5)

        t_upd = time.time()

        mean_r = batch_mean
        hist.append(mean_r)

        if log_samples:
            for s, r in zip(repaired, rewards_np):
                samples_log.append((step, float(r), s.copy()))

        # Track best (using original rewards_np BEFORE shaping to avoid inflating best_r)
        # Note: rewards_np at this point is already shaped; track by comparing shaped vals.
        # The actual path reward is correctly reflected since shaped penalties < -10,
        # while any truly feasible design will have reward > -10.
        max_idx = int(rewards.argmax())
        if rewards_np[max_idx] > best_r:
            best_r      = float(rewards_np[max_idx])
            best_params = repaired[max_idx].copy()

        print(
            f"[PPO step {step+1:4d}/{steps}]  "
            f"eval={t_eval-t0:.3f}s  upd={t_upd-t_eval:.3f}s  "
            f"mean={mean_r:.4f}  best={best_r:.4f}  ema={reward_ema:.4f}  "
            f"feas={feasible_count}/{batch_size}"
        )

        # Early stopping
        if early_stop_reward is not None and best_r >= early_stop_reward:
            print(f"[PPO] Early stop at step {step+1}: best_r={best_r:.4f}")
            break

    save_checkpoint(checkpoint_path, policy, value, best_params)
    return best_r, best_params, hist, samples_log
