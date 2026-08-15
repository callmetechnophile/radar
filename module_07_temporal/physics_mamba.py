"""PhotonShield AI — Module 07 Physics-Aware Mamba Temporal Model for Oxford Radar."""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_07_temporal.mamba_temporal import OxfordMambaTemporalModel, MiniMambaBlock


class OxfordPhysicsHead(nn.Module):
    """Lightweight MLP predicting planar rigid-body kinematics from temporal latent features.

    Predicts:
    1. dx, dy (inter-frame displacement in meters)
    2. vx, vy (linear forward/lateral velocity in m/s)
    3. omega (yaw angular velocity in rad/s)
    Total output dimension K = 5.
    """

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 32, num_outputs: int = 5) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting kinematic quantities [B, T, 5]."""
        return self.net(z)


class OxfordPhysicsAwareMamba(nn.Module):
    """Complete Physics-Aware Deterministic Mamba Temporal Model.

    Combines:
    1. Deterministic Mamba Temporal Sequence Inpainter (B2 architecture)
    2. Auxiliary Kinematic / Motion Prediction Physics Head
    3. Physics-informed multi-task regularization during training.

    Inference Guarantee: RADAR ONLY. Physics observables are strictly used during training.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        mamba_layers: int = 2,
        physics_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 1. Base Mamba Temporal Encoder & Reconstruction Backbone
        self.in_proj = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList([
            MiniMambaBlock(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(mamba_layers)
        ])

        self.recon_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # 2. Auxiliary Kinematics Physics Head
        self.physics_head = OxfordPhysicsHead(
            feature_dim=hidden_dim,
            hidden_dim=physics_hidden_dim,
            num_outputs=5,
        )

    def forward_encoder(
        self,
        x_corr: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract latent representation, reconstructed radar features, and predicted physics observables.

        Args:
            x_corr: Corrupted sequence [B, T, D] with missing frames zeroed.
            mask: Binary mask [B, T, 1] (1=observed, 0=missing).

        Returns:
            Tuple of (latent features [B, T, hidden_dim], reconstructed radar [B, T, D], physics [B, T, 5]).
        """
        B, T, D = x_corr.shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        in_feat = torch.cat([x_corr, mask], dim=-1)
        h = self.in_proj(in_feat)

        for layer in self.layers:
            h = layer(h)

        recon_out = self.recon_head(h)
        # Physics head branches directly from the internal temporal latent state
        phys_out = self.physics_head(h)

        return h, recon_out, phys_out

    def forward(
        self,
        x_corr: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic reconstruction forward pass (Radar only)."""
        _, recon_out, _ = self.forward_encoder(x_corr, mask)
        return recon_out

    def reconstruct(
        self,
        x_corr: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Standard inpainting interface: retains observed frames, replaces missing frames."""
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        pred = self.forward(x_corr, mask)
        return x_corr * mask + pred * (1.0 - mask)

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        mask: torch.Tensor,
        physics_targets: torch.Tensor,
        lambda_phys: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined reconstruction and physical regularization loss.

        Args:
            x_clean: Ground-truth clean radar feature sequence [B, T, D].
            mask: Binary mask [B, T, 1].
            physics_targets: Ground-truth kinematic vectors [B, T, 5] (dx, dy, vx, vy, yaw_rate).
            lambda_phys: Physics loss weight.

        Returns:
            Tuple of (total loss tensor, loss breakdown dictionary).
        """
        B, T, D = x_clean.shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        x_corr = x_clean * mask
        _, recon_pred, phys_pred = self.forward_encoder(x_corr, mask)

        # 1. Missing-region focused reconstruction loss
        unobs_loss = F.mse_loss(recon_pred * (1.0 - mask), x_clean * (1.0 - mask))
        obs_loss = F.mse_loss(recon_pred * mask, x_clean * mask)
        l_rec = unobs_loss * 3.0 + obs_loss

        # 2. Physics losses
        # Displacement loss: dx, dy
        l_disp = F.smooth_l1_loss(phys_pred[:, :, :2], physics_targets[:, :, :2])
        # Velocity loss: vx, vy
        l_vel = F.smooth_l1_loss(phys_pred[:, :, 2:4], physics_targets[:, :, 2:4])
        # Angular velocity loss: omega
        l_yaw = F.smooth_l1_loss(phys_pred[:, :, 4:], physics_targets[:, :, 4:])

        l_phys = l_disp + l_vel + 0.5 * l_yaw

        l_total = l_rec + lambda_phys * l_phys

        return l_total, {
            "loss_total": float(l_total.item()),
            "loss_rec": float(l_rec.item()),
            "loss_phys": float(l_phys.item()),
            "loss_disp": float(l_disp.item()),
            "loss_vel": float(l_vel.item()),
            "loss_yaw": float(l_yaw.item()),
        }
