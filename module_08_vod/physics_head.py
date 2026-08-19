"""Kinematic Physics Head and Differentiable Kinematic Consistency Loss for VoD."""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_08_vod.constants import DT_NOMINAL


class VoDPhysicsHead(nn.Module):
    """Predicts rigid-body planar kinematics and radial physical observables from latent state z.

    Predicts:
    1. dx, dy (inter-frame displacement in meters)
    2. vx, vy (linear velocity in m/s)
    3. omega (yaw angular velocity in rad/s)
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 32, num_outputs: int = 5) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass returning [B, T, 5] kinematic predictions."""
        return self.net(z)


class VoDPhysicsLoss(nn.Module):
    """Differentiable kinematic regularizer enforcing physical consistency on predicted trajectories."""

    def __init__(
        self,
        dt: float = DT_NOMINAL,
        lambda_disp: float = 1.0,
        lambda_acc: float = 0.1,
        a_ref: float = 6.0,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.lambda_disp = float(lambda_disp)
        self.lambda_acc = float(lambda_acc)
        self.a_ref = float(a_ref)

    def forward(
        self,
        pred_kinematics: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute kinematic consistency loss from [B, T, 5] predictions.

        Indices: 0: dx, 1: dy, 2: vx, 3: vy, 4: omega
        """
        dx = pred_kinematics[:, :, 0]
        dy = pred_kinematics[:, :, 1]
        vx = pred_kinematics[:, :, 2]
        vy = pred_kinematics[:, :, 3]

        # 1. Displacement consistency: dx ≈ vx * dt, dy ≈ vy * dt
        disp_res_x = dx - vx * self.dt
        disp_res_y = dy - vy * self.dt
        l_disp = F.smooth_l1_loss(disp_res_x, torch.zeros_like(disp_res_x)) + F.smooth_l1_loss(disp_res_y, torch.zeros_like(disp_res_y))

        # 2. Acceleration bounds: soft penalty on excessive da/dt
        if pred_kinematics.shape[1] > 1:
            ax = (vx[:, 1:] - vx[:, :-1]) / self.dt
            ay = (vy[:, 1:] - vy[:, :-1]) / self.dt
            a_mag = torch.sqrt(ax**2 + ay**2 + 1e-6)
            l_acc = torch.mean(F.softplus(a_mag - self.a_ref))
        else:
            l_acc = torch.tensor(0.0, device=pred_kinematics.device)

        total_loss = self.lambda_disp * l_disp + self.lambda_acc * l_acc

        components = {
            "loss_physics_total": total_loss,
            "loss_displacement": l_disp,
            "loss_acceleration": l_acc,
        }
        return total_loss, components
