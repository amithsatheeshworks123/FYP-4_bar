"""
Stateless environment for path-generation task.
Design vector p = [L1,L2,L3,L4,xO2,yO2], reward from path_kinematics.path_reward.
"""

import numpy as np
from v2.path_kinematics import path_reward


class PathEnv:
    def __init__(self, bounds):
        self.bounds = bounds
        self.low = np.array([b[0] for b in bounds])
        self.high = np.array([b[1] for b in bounds])

    def step(self, action, target_line=None):
        action = np.clip(np.array(action, dtype=float), self.low, self.high)
        r, line, pts, mse, max_dev = path_reward(action, target_line=target_line)
        return action, r, line, pts
