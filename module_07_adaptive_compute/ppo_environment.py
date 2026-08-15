"""Reinforcement Learning Environment for Radar Adaptive Diffusion Compute.

Treats each radar sequence as a contextual bandit / 1-step MDP episode:
1. Reset with radar sequence and temporal corruption.
2. Observe 9-dimensional state vector.
3. Step with action in {0, 1, 2, 3} (corresponding to 5, 10, 20, 50 diffusion steps).
4. Run frozen V2 inpainting and perception.
5. Return reward and episode termination.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import torch
import torch.nn.functional as F

from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_05_latent_diffusion.denoiser import LightweightDenoiser
from module_05_latent_diffusion.scheduler import DDPMScheduler
from module_05_latent_diffusion.corruption import RadarLatentCorruption
from module_06_physics.latent_physics_head import LatentPhysicsHead
from module_06_physics.physics_losses import RadarPhysicsLoss
from module_07_adaptive_compute.action_space import ACTIONS, IDX_TO_ACTION
from module_07_adaptive_compute.state_encoder import AdaptiveComputeStateEncoder
from module_07_adaptive_compute.ppo_reward import PPORewardCalculator


class RadarAdaptiveComputeEnv:
    """Radar Adaptive Diffusion Compute RL Environment."""

    def __init__(
        self,
        encoder: PhotonV0,
        denoiser: LightweightDenoiser,
        physics_head: LatentPhysicsHead,
        scheduler: DDPMScheduler,
        physics_loss: RadarPhysicsLoss,
        state_encoder: AdaptiveComputeStateEncoder,
        reward_calculator: PPORewardCalculator,
        device: torch.device,
        corruption_prob: float = 0.20,
    ) -> None:
        self.encoder = encoder
        self.denoiser = denoiser
        self.physics_head = physics_head
        self.scheduler = scheduler
        self.physics_loss = physics_loss
        self.state_encoder = state_encoder
        self.reward_calculator = reward_calculator
        self.device = device
        self.corruption_prob = float(corruption_prob)
        self.corr_op = RadarLatentCorruption({
            "enabled": True,
            "frame_dropout": {"enabled": True, "probability": self.corruption_prob}
        })

        self.current_x: Optional[torch.Tensor] = None
        self.current_y: Optional[torch.Tensor] = None
        self.current_z0: Optional[torch.Tensor] = None
        self.current_zc: Optional[torch.Tensor] = None
        self.current_mask: Optional[torch.Tensor] = None
        self.current_state: Optional[torch.Tensor] = None
        self.current_baseline_l_perc: Optional[float] = None

    def reset_with_sample(
        self,
        x_clean: torch.Tensor,
        y_cls: torch.Tensor,
        baseline_l_perc: Optional[float] = None,
    ) -> torch.Tensor:
        """Reset environment with a specific radar sequence batch.

        Args:
            x_clean: Clean input feature sequence `[1, 16, 64]`.
            y_cls: Target classification class `[1]`.
            baseline_l_perc: Optional precomputed 50-step anchor perception loss.

        Returns:
            Initial state tensor `[1, 9]`.
        """
        self.current_x = x_clean.to(self.device)
        self.current_y = y_cls.to(self.device)

        with torch.no_grad():
            self.current_z0, _ = self.encoder.extract_latents(self.current_x)
            self.current_zc, self.current_mask = self.corr_op(self.current_z0)
            self.current_state, _ = self.state_encoder(self.current_zc, self.current_mask)
            self.current_baseline_l_perc = baseline_l_perc

        return self.current_state

    def step(
        self,
        action_idx: int,
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Execute selected action (diffusion steps) on the current sequence.

        Args:
            action_idx: Action index in 0..3 (mapping to 5, 10, 20, 50 steps).

        Returns:
            Tuple of (next_state, reward, done, info_dict).
        """
        num_steps = IDX_TO_ACTION[action_idx]

        with torch.no_grad():
            # 1. Run Frozen V2 Diffusion Reconstruction
            zh = self.scheduler.reconstruct(
                self.denoiser,
                self.current_zc,
                self.current_mask,
                num_inference_steps=num_steps,
                deterministic=True,
            )

            # 2. Perception
            logits = self.encoder.classification_head(zh[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            pred_cls = int(torch.argmax(probs, dim=-1).item())
            l_perc = float(F.cross_entropy(logits, self.current_y).item())

            # 3. Physics
            l_phys, p_comp = self.physics_loss(zh)
            l_phys_val = float(l_phys.item())
            kin_res = float(torch.mean(torch.abs(p_comp["kin_residual"])).item())

            obs_pred = self.physics_head(zh)
            r_gt = self.physics_loss.raw_extractor.extract_range(self.current_x[..., 0:30])
            v_gt = self.physics_loss.raw_extractor.extract_velocity(self.current_x[..., 30:60])
            r_mae = float(torch.mean(torch.abs(obs_pred["range"] - r_gt)).item())
            v_mae = float(torch.mean(torch.abs(obs_pred["velocity"] - v_gt)).item())

            # 4. Reconstruction MSE
            diff_sq = (zh - self.current_z0) ** 2
            full_mse = float(torch.mean(diff_sq).item())
            miss_mask = 1.0 - self.current_mask
            miss_cnt = torch.sum(miss_mask)
            miss_mse = float((torch.sum(diff_sq * miss_mask) / (miss_cnt * 64)).item()) if miss_cnt > 0 else 0.0

            # 5. Compute Reward
            reward, breakdown = self.reward_calculator.compute_reward(
                l_perc_selected=l_perc,
                l_phys_selected=l_phys_val,
                action_idx=action_idx,
                l_perc_baseline_50=self.current_baseline_l_perc,
            )

        info = {
            "action_idx": action_idx,
            "diffusion_steps": num_steps,
            "pred_cls": pred_cls,
            "true_cls": int(self.current_y.item()),
            "correct": 1 if pred_cls == int(self.current_y.item()) else 0,
            "l_perc": l_perc,
            "l_phys": l_phys_val,
            "kin_res": kin_res,
            "r_mae": r_mae,
            "v_mae": v_mae,
            "miss_mse": miss_mse,
            "full_mse": full_mse,
            "reward_breakdown": breakdown,
        }

        # 1-step episode
        done = True
        next_state = self.current_state

        return next_state, reward, done, info
