"""Lightweight Shared Point-Wise Radar Encoder for VoD 3D Perception."""

from __future__ import annotations

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_08_vod.constants import RADAR_POINT_CHANNELS, POINT_EMBED_DIM


class RadarPointEncoder(nn.Module):
    """Permutation-invariant point-cloud neural encoder for native VoD radar.

    Architecture:
        Input point features: [B, N, 7] or [N, 7]
            ↓
        Linear(7 -> hidden_dim) + LayerNorm + SiLU
            ↓
        Linear(hidden_dim -> out_dim) + LayerNorm + SiLU
            ↓
        Permutation-Invariant Pooling (Max Pooling)
            ↓
        Frame Token [B, out_dim=64]
    """

    def __init__(
        self,
        in_channels: int = RADAR_POINT_CHANNELS,
        hidden_dim: int = 32,
        out_dim: int = POINT_EMBED_DIM,
        pooling: str = "max",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.pooling = pooling.lower()

        if self.pooling not in ("max", "mean"):
            raise ValueError(f"Unsupported pooling mode: {pooling}. Use 'max' or 'mean'.")

        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode a point cloud into a 64-D frame token.

        Args:
            points: Tensor of shape `[N, 7]` or `[B, N, 7]`.

        Returns:
            Tensor of shape `[64]` or `[B, 64]`.
        """
        is_unbatched = points.dim() == 2
        if is_unbatched:
            points = points.unsqueeze(0)  # [1, N, 7]

        B, N, C = points.shape
        if N == 0:
            # Handle empty point cloud
            out = torch.zeros((B, self.out_dim), device=points.device, dtype=points.dtype)
            return out.squeeze(0) if is_unbatched else out

        # Point-wise MLP
        point_feats = self.point_mlp(points)  # [B, N, out_dim]

        # Permutation-invariant aggregation across points dimension
        if self.pooling == "max":
            frame_token, _ = torch.max(point_feats, dim=1)  # [B, out_dim]
        else:
            frame_token = torch.mean(point_feats, dim=1)  # [B, out_dim]

        if is_unbatched:
            frame_token = frame_token.squeeze(0)

        return frame_token
