"""
Gym-like six-step sequential environment for four-bar linkage design.

The agent selects ONE design parameter per step:
  step 0 → L1   (frame / ground link)
  step 1 → L2   (crank)
  step 2 → L3   (coupler)
  step 3 → L4   (rocker)
  step 4 → xO2  (crank pivot x)
  step 5 → yO2  (crank pivot y)

Action space : 1-D continuous ∈ [-1, 1], linearly mapped to the current
               parameter's bounds before being stored.

Observation (15-D):
  [0:6]   normalised params (each ∈ [0, 1] via min-max of its bound)
  [6:12]  one-hot encoding of the current step index (all-zero at terminal)
  [12:15] feasibility signals:
            [0] l2_shortest  – 1 if L2 < all other link lengths (valid from step 2)
            [1] grashof_ok   – 1 if Grashof crank-rocker holds (valid from step 4)
            [2] reserved / always 0.0 (path validity only known after step 5,
                                       which is terminal so no next obs is used)

Intermediate rewards:
  step 1 (L2 chosen) : +0.5 if L2 < L1,   -0.5 otherwise
  step 3 (L4 chosen) : +1.0 if full Grashof crank-rocker satisfied,
                        else negative penalty ∝ violation magnitude
  step 5 (yO2 chosen): float — raw batch_path_reward for the full design
  all other steps    : 0.0

BOUNDS (matches app.py):
  [(0.05, 0.4)] * 4 + [(-0.1, 0.2), (-0.1, 0.2)]
"""

import numpy as np
from path_kinematics import batch_path_reward


class FourBarEnv:
    """
    Stateless, deterministic, episodic six-step environment.

    Each episode is exactly N_STEPS = 6 transitions.  There is no stochasticity
    in the environment itself — all randomness comes from the policy.

    Parameters
    ----------
    bounds      : list of (low, high) pairs, length 6
    target_line : (a, b) tuple or None — passed to batch_path_reward
    """

    PARAM_NAMES = ['L1', 'L2', 'L3', 'L4', 'xO2', 'yO2']
    N_PARAMS    = 6
    N_STEPS     = 6
    OBS_DIM     = 15   # 6 params + 6 step one-hot + 3 feasibility signals
    ACT_DIM     = 1    # scalar action in [-1, 1]

    def __init__(self, bounds, target_line=None, target_pts=None):
        self.bounds      = list(bounds)
        self.target_line = target_line
        self.target_pts  = target_pts

        self.lows   = np.array([b[0] for b in bounds], dtype=np.float32)
        self.highs  = np.array([b[1] for b in bounds], dtype=np.float32)
        self.mids   = 0.5 * (self.lows + self.highs)
        self.ranges = self.highs - self.lows  # for normalisation

        # episode state (initialised by reset)
        self.params   = None
        self.step_idx = None
        self.done     = None

    # ── Public interface ───────────────────────────────────────────────────────

    def reset(self):
        """
        Begin a new episode.  All params are initialised to their bound midpoints.
        Returns the initial 15-D observation.
        """
        self.params   = self.mids.copy()
        self.step_idx = 0
        self.done     = False
        return self._obs()

    def step(self, action):
        """
        Apply one action (scalar float in [-1, 1]).

        Parameters
        ----------
        action : float — agent's choice in [-1, 1]

        Returns
        -------
        obs    : np.ndarray (15,) — next observation
        reward : float
        done   : bool — True after step 5
        info   : dict with keys 'step' (completed step index) and 'params'
        """
        if self.done:
            raise RuntimeError("Episode finished — call reset() first.")

        k         = self.step_idx
        low, high = self.bounds[k]

        # Map action linearly from [-1, 1] → [low, high]
        param_val       = float(np.clip(low + (float(action) + 1.0) * 0.5 * (high - low),
                                        low, high))
        self.params[k]  = param_val

        reward         = self._reward(k)
        self.step_idx += 1
        done           = (self.step_idx == self.N_STEPS)
        self.done      = done

        obs  = self._obs()
        info = {"step": k, "params": self.params.copy()}
        return obs, reward, done, info

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def obs_dim(self):
        return self.OBS_DIM

    @property
    def act_dim(self):
        return self.ACT_DIM

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reward(self, step_completed):
        """
        Compute the reward emitted after completing step `step_completed`.

        Intermediate rewards at steps 1 and 3 guide the agent toward Grashof
        feasibility before the expensive kinematics evaluation at step 5.
        """
        p = self.params

        if step_completed == 1:
            # L2 has just been chosen.  +0.5 if it is shorter than L1 (crank-rocker
            # requires L2 to be the shortest link).
            return 0.5 if p[1] < p[0] else -0.5

        elif step_completed == 3:
            # All four link lengths are now known.  Check full Grashof crank-rocker:
            #   s + l <= p + q  and  argmin(arr4) == 1  (L2 must be shortest)
            arr4          = p[:4]
            s             = float(arr4.min())
            l             = float(arr4.max())
            others_sum    = float(arr4.sum()) - s - l
            grashof_viol  = max(0.0, (s + l) - others_sum)
            l2_is_min     = (arr4.argmin() == 1)

            if grashof_viol == 0.0 and l2_is_min:
                return 1.0  # fully feasible — maximum intermediate reward
            else:
                # Penalty proportional to violation; kept in [-5, -1] so it
                # doesn't overwhelm the final path reward.
                l2_penalty = 0.0 if l2_is_min else 0.5
                return -(1.0 + 3.0 * grashof_viol + l2_penalty)

        elif step_completed == 5:
            # Full design is now complete — evaluate actual path quality.
            if self.target_pts is not None and self.target_line is None:
                from target_trajectory import trajectory_reward
                from path_kinematics import coupler_path, is_crank_rocker_with_L2_crank
                L1, L2, L3, L4 = self.params[:4]
                if not is_crank_rocker_with_L2_crank(L1, L2, L3, L4):
                    return -1e6
                pts, valid_frac, all_valid = coupler_path(self.params)
                if not all_valid or valid_frac < 0.99:
                    return -1e6
                return trajectory_reward(pts, self.target_pts)
            r = batch_path_reward(
                self.params[np.newaxis, :],
                target_line=self.target_line
            )
            return float(r[0])

        else:
            return 0.0

    def _obs(self):
        """
        Build the 15-D observation vector.

          [0:6]   Params normalised to [0, 1] via min-max scaling of bounds.
                  Un-chosen params remain at their bound midpoints (0.5 normalised).
          [6:12]  One-hot encoding of the NEXT step to be taken.
                  All zeros at the terminal state (step_idx == N_STEPS).
          [12:15] Feasibility signals computed from params chosen so far:
                    [12] l2_shortest – 1.0 if L2 < all other link lengths
                                       (meaningful once steps 0 & 1 are done)
                    [13] grashof_ok  – 1.0 if Grashof crank-rocker condition holds
                                       (meaningful once steps 0-3 are done)
                    [14] always 0.0  (path validity only computed at terminal step,
                                       which is the done step — no next obs needed)
        """
        # ── Normalised params ──────────────────────────────────────────────────
        params_norm = ((self.params - self.lows) / (self.ranges + 1e-8)).astype(np.float32)

        # ── Step one-hot ───────────────────────────────────────────────────────
        step_oh = np.zeros(self.N_STEPS, dtype=np.float32)
        if self.step_idx < self.N_STEPS:
            step_oh[self.step_idx] = 1.0

        # ── Feasibility signals ────────────────────────────────────────────────
        p = self.params

        # Signal 0: L2 is the shortest of {L1, L2, L3, L4}.
        # Only meaningful after both L1 (step 0) and L2 (step 1) have been set,
        # i.e., once step_idx >= 2.
        if self.step_idx >= 2:
            arr4        = p[:4]
            l2_shortest = float(arr4.argmin() == 1)
        else:
            l2_shortest = 0.0

        # Signal 1: Grashof crank-rocker condition holds for L1..L4.
        # Only meaningful after all four link lengths are set (step_idx >= 4).
        if self.step_idx >= 4:
            arr4       = p[:4]
            s          = float(arr4.min())
            l_val      = float(arr4.max())
            others_sum = float(arr4.sum()) - s - l_val
            grashof_ok = float(
                (arr4.argmin() == 1) and (s + l_val <= others_sum + 1e-9)
            )
        else:
            grashof_ok = 0.0

        feasibility = np.array([l2_shortest, grashof_ok, 0.0], dtype=np.float32)

        return np.concatenate([params_norm, step_oh, feasibility])
