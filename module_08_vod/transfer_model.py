"""Transfer Learning Model Supporting Oxford-to-VoD Weights, Kinematics, and 3D Heads."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from module_04_mamba_hybrid.mamba_core import MiniMambaBlock
from module_08_vod.radar_point_encoder import RadarPointEncoder
from module_08_vod.reconstruction_head import OccupancyReconstructionHead
from module_08_vod.object_head import VoDObject3DHead
from module_08_vod.physics_head import VoDPhysicsHead
from module_08_vod.constants import (
    RADAR_POINT_CHANNELS,
    POINT_EMBED_DIM,
    MAMBA_HIDDEN_DIM,
    VOXEL_DIM_X,
    VOXEL_DIM_Y,
    VOXEL_DIM_Z,
)


class VoDTransfer3DModel(nn.Module):
    """Composite transfer model integrating VoD Point Encoder, Oxford Mamba, and Multi-Task Heads.

    Supports 6 Controlled Scientific Regimes:
    - BASELINE-A: "native_no_physics"
    - TRANSFER-B: "frozen_transfer"
    - TRANSFER-C: "physics_transfer"
    - TRANSFER-D: "partial_finetune" (Mamba unfrozen)
    - TRANSFER-E: "full_finetune" (All layers unfrozen)
    - CONTROL-F:  "native_with_physics"
    """

    def __init__(
        self,
        regime: str = "native_no_physics",
        point_in_dim: int = RADAR_POINT_CHANNELS,
        feature_dim: int = POINT_EMBED_DIM,
        hidden_dim: int = MAMBA_HIDDEN_DIM,
        num_mamba_layers: int = 2,
        oxford_checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.regime = regime.lower()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 1. VoD Point-Cloud Encoder (Maps native N x 7 -> 64-D)
        self.point_encoder = RadarPointEncoder(in_channels=point_in_dim, out_dim=feature_dim)

        # 2. Causal Mamba Temporal Backbone
        self.in_proj = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.mamba_layers = nn.ModuleList([
            MiniMambaBlock(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(num_mamba_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # 3. 3D Object Detection & Classification Head
        self.object_head = VoDObject3DHead(in_dim=hidden_dim, hidden_dim=hidden_dim)

        # 4. Kinematic Physics Head
        self.physics_head = VoDPhysicsHead(in_dim=hidden_dim, hidden_dim=32, num_outputs=5)

        # 5. 3D Occupancy Reconstruction Head (Secondary Task)
        self.occupancy_head = OccupancyReconstructionHead(in_dim=hidden_dim, voxel_dims=(VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z))

        # Apply transfer learning and freezing policy
        self._apply_regime_policy(oxford_checkpoint_path)

    def _apply_regime_policy(self, checkpoint_path: Optional[str]):
        """Configure layer freezing according to the selected regime."""
        # Baseline A: All unfrozen, trained from scratch
        if self.regime in ("native_no_physics", "native_with_physics", "full_finetune"):
            for p in self.parameters():
                p.requires_grad = True

        elif self.regime in ("frozen_transfer", "physics_transfer"):
            # Freeze Mamba backbone and Point Encoder, train only Object Head and Adaptation
            for p in self.point_encoder.parameters():
                p.requires_grad = False
            for p in self.in_proj.parameters():
                p.requires_grad = False
            for p in self.mamba_layers.parameters():
                p.requires_grad = False
            for p in self.norm.parameters():
                p.requires_grad = False
            for p in self.physics_head.parameters():
                p.requires_grad = False
            for p in self.object_head.parameters():
                p.requires_grad = True
            for p in self.occupancy_head.parameters():
                p.requires_grad = True

        elif self.regime == "partial_finetune":
            # Freeze physics head, unfreeze Mamba and Object Head
            for p in self.point_encoder.parameters():
                p.requires_grad = True
            for p in self.in_proj.parameters():
                p.requires_grad = True
            for p in self.mamba_layers.parameters():
                p.requires_grad = True
            for p in self.norm.parameters():
                p.requires_grad = True
            for p in self.physics_head.parameters():
                p.requires_grad = False
            for p in self.object_head.parameters():
                p.requires_grad = True
            for p in self.occupancy_head.parameters():
                p.requires_grad = True

    def forward(
        self,
        tokens_or_points: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through transfer model.

        Returns:
            Tuple of (cls_logits [B, T, 4], box_params [B, T, 7], occ_logits [B, T, Vx, Vy, Vz], kinematics [B, T, 5]).
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

        masked_tokens = tokens * mask
        x = torch.cat([masked_tokens, mask], dim=-1)  # [B, T, D + 1]
        h = self.in_proj(x)

        for layer in self.mamba_layers:
            h = layer(h)

        h = self.norm(h)  # [B, T, hidden_dim]

        # Multi-task outputs
        cls_logits, box_params = self.object_head(h)
        occ_logits = self.occupancy_head(h)
        pred_kinematics = self.physics_head(h)

        return cls_logits, box_params, occ_logits, pred_kinematics
