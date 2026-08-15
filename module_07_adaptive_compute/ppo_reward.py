"""Reward and objective formulation for PhotonShield V3.3 PPO Controller.

Defines:
    J = alpha * L_perc + beta * L_phys + gamma_compute * (N / 50)
    R = -J - eta * SafetyPenalty
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_07_adaptive_compute.action_space import IDX_TO_ACTION


class PPORewardCalculator:
    """Calculates scalar downstream reward for PPO training with perception safety penalty."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.25,
        gamma_compute: float = 0.10,
        eta_safety: float = 0.50,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma_compute = float(gamma_compute)
        self.eta_safety = float(eta_safety)

    def compute_reward(
        self,
        l_perc_selected: float,
        l_phys_selected: float,
        action_idx: int,
        l_perc_baseline_50: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate scalar PPO reward.

        Args:
            l_perc_selected: Cross-entropy perception loss for chosen action.
            l_phys_selected: Physics loss for chosen action.
            action_idx: Action index in 0..3 (mapping to 5, 10, 20, 50).
            l_perc_baseline_50: Optional 50-step reference perception loss for safety penalty.

        Returns:
            Tuple of (scalar_reward, metric_breakdown_dict).
        """
        diffusion_steps = IDX_TO_ACTION[action_idx]
        compute_cost = float(diffusion_steps) / 50.0

        # Primary objective: J(N)
        j_obj = (self.alpha * l_perc_selected) + (self.beta * l_phys_selected) + (self.gamma_compute * compute_cost)

        # Safety penalty: penalizes perception loss exceeding the 50-step anchor
        safety_penalty = 0.0
        if l_perc_baseline_50 is not None:
            deg = max(0.0, l_perc_selected - l_perc_baseline_50)
            safety_penalty = self.eta_safety * deg

        reward = -j_obj - safety_penalty

        breakdown = {
            "reward": float(reward),
            "j_objective": float(j_obj),
            "l_perc": float(l_perc_selected),
            "l_phys": float(l_phys_selected),
            "compute_cost": float(compute_cost),
            "safety_penalty": float(safety_penalty),
            "steps": diffusion_steps,
        }

        return float(reward), breakdown
