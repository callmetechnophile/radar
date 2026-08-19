"""Lightweight 3D Occupancy Reconstruction Head."""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn

from module_08_vod.constants import (
    POINT_EMBED_DIM,
    VOXEL_DIM_X,
    VOXEL_DIM_Y,
    VOXEL_DIM_Z,
    TOTAL_VOXELS,
)


class OccupancyReconstructionHead(nn.Module):
    """Decodes a compact 64-D latent representation into 3D voxel occupancy logits.

    Architecture:
        Latent Token [B, 64]
            ↓
        Linear(64 -> 128) + LayerNorm + SiLU
            ↓
        Linear(128 -> 512) + LayerNorm + SiLU
            ↓
        Linear(512 -> Total_Voxels = 8,192)
            ↓
        Reshape to [B, Vx=32, Vy=32, Vz=8]
    """

    def __init__(
        self,
        in_dim: int = POINT_EMBED_DIM,
        hidden_dim1: int = 128,
        hidden_dim2: int = 512,
        voxel_dims: Tuple[int, int, int] = (VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z),
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.voxel_dims = voxel_dims
        self.total_voxels = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]

        self.decoder_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.SiLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.SiLU(),
            nn.Linear(hidden_dim2, self.total_voxels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector into 3D occupancy logits.

        Args:
            z: Latent tensor of shape `[B, in_dim]` or `[B, T, in_dim]`.

        Returns:
            Occupancy logits of shape `[B, Vx, Vy, Vz]` or `[B, T, Vx, Vy, Vz]`.
        """
        vx, vy, vz = self.voxel_dims
        if z.dim() == 2:
            B, D = z.shape
            logits_flat = self.decoder_mlp(z)  # [B, total_voxels]
            return logits_flat.view(B, vx, vy, vz)
        elif z.dim() == 3:
            B, T, D = z.shape
            z_flat = z.view(B * T, D)
            logits_flat = self.decoder_mlp(z_flat)  # [B*T, total_voxels]
            return logits_flat.view(B, T, vx, vy, vz)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {z.shape}")
