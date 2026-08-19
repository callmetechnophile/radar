"""VoD Frame-Wise Baseline (A) and VoD Mamba Temporal Model (B)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from module_04_mamba_hybrid.mamba_core import MiniMambaBlock
from module_08_vod.radar_point_encoder import RadarPointEncoder
from module_08_vod.reconstruction_head import OccupancyReconstructionHead
from module_08_vod.constants import (
    RADAR_POINT_CHANNELS,
    POINT_EMBED_DIM,
    MAMBA_HIDDEN_DIM,
    VOXEL_DIM_X,
    VOXEL_DIM_Y,
    VOXEL_DIM_Z,
)


class VoDFramewiseBaseline(nn.Module):
    """Experiment A: Controlled Non-Temporal Frame-Wise Baseline.

    Processes each radar frame independently:
    Point Cloud [B, T, N, 7] -> Point Encoder -> [B, T, 64] -> Frame MLP -> Occupancy [B, T, Vx, Vy, Vz]
    """

    def __init__(
        self,
        point_in_dim: int = RADAR_POINT_CHANNELS,
        feature_dim: int = POINT_EMBED_DIM,
        voxel_dims: Tuple[int, int, int] = (VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z),
    ) -> None:
        super().__init__()
        self.point_encoder = RadarPointEncoder(in_channels=point_in_dim, out_dim=feature_dim)
        self.reconstruction_head = OccupancyReconstructionHead(in_dim=feature_dim, voxel_dims=voxel_dims)

    def forward(
        self,
        tokens_or_points: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for frame-wise baseline.

        Args:
            tokens_or_points: Pre-encoded tokens `[B, T, 64]` or point tensor `[B, T, N, 7]`.
            mask: Optional observation mask `[B, T, 1]`.

        Returns:
            Occupancy logits `[B, T, Vx, Vy, Vz]`.
        """
        if tokens_or_points.dim() == 4:
            # [B, T, N, 7] -> encode per frame
            B, T, N, C = tokens_or_points.shape
            pts_flat = tokens_or_points.view(B * T, N, C)
            tokens = self.point_encoder(pts_flat).view(B, T, -1)
        else:
            # [B, T, 64]
            tokens = tokens_or_points

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            tokens = tokens * mask

        logits = self.reconstruction_head(tokens)  # [B, T, Vx, Vy, Vz]
        return logits


class VoDMambaTemporalModel(nn.Module):
    """Experiment B: Temporal Mamba 3D Radar Representation Model.

    Processes sequences of radar frame tokens using Causal Selective SSM:
    Input tokens [B, T, 64] + Mask [B, T, 1]
        ↓
    Input Projection (65 -> hidden_dim) + LayerNorm + SiLU
        ↓
    Mamba SSM Block 1 (Selective SSM + Causal Conv1D + SiLU + Residual)
        ↓
    Mamba SSM Block 2 (Selective SSM + Causal Conv1D + SiLU + Residual)
        ↓
    Temporal Latent Sequence [B, T, 64]
        ↓
    Reconstruction Head -> Occupancy Logits [B, T, Vx, Vy, Vz]
    """

    def __init__(
        self,
        point_in_dim: int = RADAR_POINT_CHANNELS,
        feature_dim: int = POINT_EMBED_DIM,
        hidden_dim: int = MAMBA_HIDDEN_DIM,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        voxel_dims: Tuple[int, int, int] = (VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z),
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.point_encoder = RadarPointEncoder(in_channels=point_in_dim, out_dim=feature_dim)

        # Input Projection (Features + 1 mask channel)
        self.in_proj = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Sequential Mamba SSM Blocks
        self.layers = nn.ModuleList([
            MiniMambaBlock(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # 3D Occupancy Reconstruction Head
        self.reconstruction_head = OccupancyReconstructionHead(in_dim=hidden_dim, voxel_dims=voxel_dims)

    def forward(
        self,
        tokens_or_points: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Mamba temporal model.

        Args:
            tokens_or_points: Pre-encoded tokens `[B, T, 64]` or point tensor `[B, T, N, 7]`.
            mask: Optional observation mask `[B, T, 1]`.

        Returns:
            Occupancy logits `[B, T, Vx, Vy, Vz]`.
        """
        if tokens_or_points.dim() == 4:
            B, T, N, C = tokens_or_points.shape
            pts_flat = tokens_or_points.view(B * T, N, C)
            tokens = self.point_encoder(pts_flat).view(B, T, -1)
        else:
            tokens = tokens_or_points

        B, T, D = tokens.shape
        if mask is None:
            mask = torch.ones((B, T, 1), device=tokens.device, dtype=tokens.dtype)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        # Mask corrupted/dropped input tokens
        masked_tokens = tokens * mask

        # Temporal feature concatenation with observation mask
        x = torch.cat([masked_tokens, mask], dim=-1)  # [B, T, D + 1]
        h = self.in_proj(x)  # [B, T, hidden_dim]

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)  # [B, T, hidden_dim]
        logits = self.reconstruction_head(h)  # [B, T, Vx, Vy, Vz]
        return logits
