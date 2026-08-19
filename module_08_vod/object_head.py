"""3D Bounding Box and Object Classification Head for View-of-Delft."""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_VOD_CLASSES = 4  # 0: Car, 1: Pedestrian, 2: Cyclist, 3: Background/Other


class VoDObject3DHead(nn.Module):
    """Predicts 3D bounding box parameters and multi-class classification from 64-D temporal latents.

    Outputs:
    1. Class Logits: [B, T, 4] (Car, Pedestrian, Cyclist, Background)
    2. 3D Box Parameters: [B, T, 7]
       - Center: [x, y, z] in meters
       - Dimensions: [l, w, h] in meters
       - Orientation: [yaw] in radians
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 64,
        num_classes: int = NUM_VOD_CLASSES,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        # Shared Feature Extractor
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Classification Sub-Head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # 3D Bounding-Box Regression Sub-Head (center [3], dims [3], yaw [1])
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting class logits and 3D bounding boxes.

        Args:
            z: Latent feature tensor `[B, T, 64]` or `[B, 64]`.

        Returns:
            Tuple of (cls_logits [B, T, num_classes], box_params [B, T, 7]).
        """
        is_unbatched_seq = z.dim() == 2
        if is_unbatched_seq:
            z = z.unsqueeze(1)  # [B, 1, 64]

        h = self.shared_mlp(z)  # [B, T, hidden_dim]
        cls_logits = self.cls_head(h)  # [B, T, 4]
        box_params = self.box_head(h)  # [B, T, 7]

        if is_unbatched_seq:
            cls_logits = cls_logits.squeeze(1)
            box_params = box_params.squeeze(1)

        return cls_logits, box_params
