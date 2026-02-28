"""
Standard PPO (actor-critic with GAE) for the six-step FourBarEnv.

Architecture
------------
Policy  (actor) : MLP [15 → 128 → 128 → 2]
                  output[:, 0] → tanh-squashed mean  ∈ (-1, 1)
                  output[:, 1] → log_std clamped to [-3, 0.5]
Value  (critic) : MLP [15 → 128 → 128 → 1]

Training
--------
Each outer step:
  1. Collect `batch_size` complete episodes (6 actions each) under the current
     policy with no_grad.  Log-probs and V-estimates are recorded at collection.
  2. Compute per-timestep GAE advantages (γ=0.99, λ=0.95) and returns.
  3. Normalise advantages to zero mean / unit variance across the whole batch.
  4. Run K=4 epochs of shuffled minibatch (size 64) PPO updates:
       actor_loss  = PPO clipped surrogate (ε=0.2)
       critic_loss = 0.5 × MSE(V_pred, returns)
       entropy     = entropy of current distribution
       loss = actor_loss + critic_loss − entropy_coef × entropy
     Grad-clip both networks to max_norm=0.5.

Why sequential formulation helps
---------------------------------
• Intermediate rewards at steps 1 (L2<L1?) and 3 (Grashof?) guide the agent
  toward feasible link-length ratios BEFORE the expensive kinematics call at
  step 5.  This avoids the "all rewards identical → zero gradient" collapse that
  the bandit PPO (ppo.py) suffers when 100% of designs are infeasible.
• The value network uses 6-step temporal structure: V(s_t) can learn to predict
  expected path-quality from partial observations — a richer signal than any
  single-step baseline.
• GAE discount (γ=0.99 over 6 steps) naturally weights the final path reward
  highest while propagating feasibility credit back to earlier choices.

Return signature
----------------
(best_reward, best_params, history, samples_log)

Matches ppo_train() so app.py can call both interchangeably.
  best_reward  – best final path_reward (step-5 reward) seen across all episodes
  best_params  – np.ndarray (6,) design params for that episode
  history      – list[float] mean final path_reward per training step
  samples_log  – list of (step, reward, params) if log_samples else None
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn

from sequential_env import FourBarEnv


# ── Networks ──────────────────────────────────────────────────────────────────

class SequentialPolicy(nn.Module):
    """
    MLP [15 → 128 → 128 → 2]: actor for the sequential env.

    The last layer outputs two values per action dimension:
      out[..., 0] → mean  (tanh-squashed to (-1, 1))
      out[..., 1] → log_std (clamped to [-3, 0.5])
    """
    def __init__(self, obs_dim: int = 15, act_dim: int = 1):
        super().__init__()
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128,     128), nn.Tanh(),
            nn.Linear(128, act_dim * 2),
        )

    def _dist_params(self, obs):
        """Return (mean, log_std) for the action distribution."""
        out     = self.net(obs)
        mean    = torch.tanh(out[..., :self.act_dim])            # (-1, 1)
        log_std = torch.clamp(out[..., self.act_dim:], -3.0, 0.5)
        return mean, log_std

    def get_action_and_logprob(self, obs):
        """
        Sample an action and compute its log-probability.

        Parameters
        ----------
        obs : Tensor (..., obs_dim)

        Returns
        -------
        action   : Tensor (..., act_dim) clamped to [-1, 1]
        log_prob : Tensor (...,)
        """
        mean, log_std = self._dist_params(obs)
        std  = torch.exp(log_std)
        eps  = torch.randn_like(mean)
        # Add Gaussian noise, then clamp to the valid action range
        action   = torch.clamp(mean + std * eps, -1.0, 1.0)
        log_prob = _gaussian_log_prob(action, mean, log_std)
        return action, log_prob

    def evaluate(self, obs, action):
        """
        Compute log-probability and per-dim entropy for stored (obs, action) pairs.

        Parameters
        ----------
        obs    : Tensor (B, obs_dim)
        action : Tensor (B, act_dim)

        Returns
        -------
        log_prob : Tensor (B,)
        entropy  : Tensor (B,)
        """
        mean, log_std = self._dist_params(obs)
        log_prob = _gaussian_log_prob(action, mean, log_std)
        entropy  = _gaussian_entropy(log_std)
        return log_prob, entropy


class SequentialValue(nn.Module):
    """MLP [15 → 128 → 128 → 1]: critic / state-value function V(s)."""
    def __init__(self, obs_dim: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128,     128), nn.Tanh(),
            nn.Linear(128,       1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ── Distribution helpers ──────────────────────────────────────────────────────

def _gaussian_log_prob(x, mean, log_std):
    """
    Sum of per-dimension Gaussian log-probs.
    x, mean, log_std : (..., D) → returns (...)
    """
    std = torch.exp(log_std)
    lp  = -0.5 * (((x - mean) / (std + 1e-8)) ** 2
                  + 2.0 * log_std
                  + math.log(2.0 * math.pi))
    return lp.sum(dim=-1)


def _gaussian_entropy(log_std):
    """
    Sum of per-dimension Gaussian entropies.
    log_std : (..., D) → returns (...)
    """
    ent = 0.5 * (1.0 + 2.0 * log_std + math.log(2.0 * math.pi))
    return ent.sum(dim=-1)


# ── GAE ───────────────────────────────────────────────────────────────────────

def _compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Generalised Advantage Estimation for a single episode.

    Parameters
    ----------
    rewards, values : lists of length T
    gamma, lam      : discount and GAE lambda

    Returns
    -------
    advantages : np.ndarray (T,)
    returns    : np.ndarray (T,)  —  advantages + values  (targets for critic)
    """
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae        = 0.0

    for t in reversed(range(T)):
        next_val = 0.0 if t == T - 1 else values[t + 1]   # episode terminal → 0
        delta    = rewards[t] + gamma * next_val - values[t]
        gae      = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ── Main training function ────────────────────────────────────────────────────

def ppo_sequential_train(
    bounds,
    steps       = 500,
    batch_size  = 256,
    lr          = 3e-4,
    seed        = None,
    target_line = None,
    log_samples = False,
):
    """
    Train a sequential PPO agent on FourBarEnv.

    Parameters
    ----------
    bounds      : list of (low, high) — six parameter bounds
    steps       : int  — number of outer training iterations
    batch_size  : int  — episodes collected per training step
    lr          : float — Adam learning rate (shared for actor and critic)
    seed        : int or None
    target_line : (a, b) tuple or None — passed to path_kinematics
    log_samples : bool — whether to populate samples_log

    Returns
    -------
    (best_reward, best_params, history, samples_log)

    Return types match ppo_train() for transparent use in app.py.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env    = FourBarEnv(bounds, target_line=target_line)
    policy = SequentialPolicy(obs_dim=env.obs_dim, act_dim=env.act_dim).to(device)
    value  = SequentialValue(obs_dim=env.obs_dim).to(device)

    # Separate optimisers — critic often benefits from slightly different lr
    opt_p = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_v = torch.optim.Adam(value.parameters(),  lr=lr)

    gamma    = 0.99
    lam      = 0.95
    clip_eps = 0.2
    n_epochs = 4
    mb_size  = 64

    best_r      = -1e9
    best_params = None
    hist        = []
    samples_log = [] if log_samples else None

    T = env.N_STEPS  # 6 steps per episode

    for step in range(steps):
        t0 = time.time()

        # ── Collect episodes ────────────────────────────────────────────────────
        # Lists grow to length (batch_size * T) = 256 * 6 = 1536 entries
        all_obs      = []
        all_actions  = []
        all_lp_old   = []
        all_adv      = []
        all_ret      = []

        final_rewards = []  # step-5 path rewards — used for history and best tracking

        policy.eval()
        value.eval()

        with torch.no_grad():
            for _ep in range(batch_size):
                obs = env.reset()

                ep_obs = []
                ep_act = []
                ep_lp  = []
                ep_rew = []
                ep_val = []

                for _t in range(T):
                    obs_t  = torch.tensor(obs, dtype=torch.float32,
                                          device=device).unsqueeze(0)   # (1, 15)
                    act, lp = policy.get_action_and_logprob(obs_t)      # (1,1), (1,)
                    v       = value(obs_t).item()                        # scalar

                    act_np = float(act.squeeze().cpu())
                    obs_next, reward, done, info = env.step(act_np)

                    ep_obs.append(obs)
                    ep_act.append([act_np])         # keep as (1,) for stacking
                    ep_lp.append(float(lp.squeeze().cpu()))
                    ep_rew.append(float(reward))
                    ep_val.append(v)

                    obs = obs_next

                # GAE for this episode
                adv_ep, ret_ep = _compute_gae(ep_rew, ep_val, gamma, lam)

                all_obs.extend(ep_obs)
                all_actions.extend(ep_act)
                all_lp_old.extend(ep_lp)
                all_adv.extend(adv_ep.tolist())
                all_ret.extend(ret_ep.tolist())

                # Step-5 reward = raw batch_path_reward for this design
                final_r = ep_rew[T - 1]
                final_p = info["params"]         # full 6-element design vector
                final_rewards.append(final_r)

                if final_r > best_r:
                    best_r      = final_r
                    best_params = final_p.copy()

                if log_samples:
                    samples_log.append((step, final_r, final_p.copy()))

        t_eval = time.time()

        # ── Convert to tensors ─────────────────────────────────────────────────
        # Pre-stack as numpy arrays first to avoid the slow list-of-arrays path
        obs_t    = torch.from_numpy(np.array(all_obs,      dtype=np.float32)).to(device)
        act_t    = torch.from_numpy(np.array(all_actions,  dtype=np.float32)).to(device)
        lp_old_t = torch.from_numpy(np.array(all_lp_old,  dtype=np.float32)).to(device)
        ret_t    = torch.from_numpy(np.array(all_ret,      dtype=np.float32)).to(device)
        adv_t    = torch.from_numpy(np.array(all_adv,      dtype=np.float32)).to(device)

        # Normalise advantages across the full batch (all episodes, all steps)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ── PPO update ─────────────────────────────────────────────────────────
        policy.train()
        value.train()

        N_total      = len(all_obs)
        # Entropy coefficient: cosine anneal from 0.05 → 0.01, never below 0.01
        entropy_coef = max(0.01,
                           0.05 * 0.5 * (1.0 + math.cos(math.pi * step / steps)))

        for _epoch in range(n_epochs):
            perm = torch.randperm(N_total, device=device)
            for start in range(0, N_total, mb_size):
                idx = perm[start:start + mb_size]

                mb_obs    = obs_t[idx]       # (mb, 15)
                mb_act    = act_t[idx]       # (mb, 1)
                mb_adv    = adv_t[idx]       # (mb,)
                mb_ret    = ret_t[idx]       # (mb,)
                mb_lp_old = lp_old_t[idx]   # (mb,)

                # Actor
                lp_new, entropy = policy.evaluate(mb_obs, mb_act)
                ratios   = torch.exp(lp_new - mb_lp_old)
                clipped  = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                actor_loss = -torch.min(ratios * mb_adv, clipped).mean()

                # Critic
                v_pred      = value(mb_obs)
                critic_loss = 0.5 * ((v_pred - mb_ret) ** 2).mean()

                entropy_bonus = entropy.mean()

                loss = actor_loss + critic_loss - entropy_coef * entropy_bonus

                opt_p.zero_grad()
                opt_v.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(value.parameters(),  max_norm=0.5)
                opt_p.step()
                opt_v.step()

        t_upd = time.time()

        mean_final_r  = float(np.mean(final_rewards))
        feasible_count = int(np.sum(np.array(final_rewards) > -1e5))
        hist.append(mean_final_r)

        print(
            f"[PPO-Seq step {step+1:4d}/{steps}]  "
            f"eval={t_eval-t0:.3f}s  upd={t_upd-t_eval:.3f}s  "
            f"mean={mean_final_r:.4f}  best={best_r:.4f}  "
            f"feas={feasible_count}/{batch_size}"
        )

    # Fallback: if no feasible design was ever found, return midpoint
    if best_params is None:
        lows        = np.array([b[0] for b in bounds], dtype=np.float32)
        highs       = np.array([b[1] for b in bounds], dtype=np.float32)
        best_params = 0.5 * (lows + highs)
        best_r      = -1e6

    return best_r, best_params, hist, samples_log
