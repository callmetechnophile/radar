"""Lightweight Multi-Object Prediction Heads for Dense VoD Radar Perception."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_OBJECT_CLASSES = 3  # 0: Car, 1: Pedestrian, 2: Cyclist


class AnchorBasedMultiObjectHead(nn.Module):
    """HEAD-1: Spatial Anchor-Based Multi-Object Prediction Head.

    Distributes K fixed spatial anchors across the forward BEV perception frustum:
    X in [4, 28] m, Y in [-12, 12] m.

    For each anchor k in 1..K:
    - Objectness confidence score: p_k in [0, 1]
    - Multi-class classification logits: c_k in R^3 (Car, Pedestrian, Cyclist)
    - 3D Bounding-Box offsets: [dx, dy, dz, dl, dw, dh, dyaw]
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 64,
        num_anchors: int = 16,
        num_classes: int = NUM_OBJECT_CLASSES,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Create fixed spatial anchor centers [K, 7] (x, y, z, l, w, h, yaw)
        anchors = []
        xs = np.linspace(6.0, 26.0, 4)
        ys = np.linspace(-9.0, 9.0, 4)
        for x in xs:
            for y in ys:
                anchors.append([x, y, -0.5, 3.5, 1.8, 1.6, 0.0])
        self.register_buffer("anchor_priors", torch.tensor(anchors, dtype=torch.float32))

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Output projection per anchor: (1 objectness + 3 classes + 7 box offsets) = 11 outputs per anchor
        self.head_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_anchors * 11),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass predicting multi-object candidates.

        Args:
            z: Latent feature tensor `[B, T, 64]` or `[B, 64]`.

        Returns:
            Tuple of:
            - confidences: `[B, T, K, 1]` (sigmoid objectness probabilities)
            - class_logits: `[B, T, K, 3]` (multi-class logits)
            - box_params: `[B, T, K, 7]` (predicted 3D coordinates in meters)
        """
        is_unbatched_seq = z.dim() == 2
        if is_unbatched_seq:
            z = z.unsqueeze(1)  # [B, 1, 64]

        B, T, _ = z.shape
        h = self.shared_mlp(z)  # [B, T, hidden_dim]
        out_flat = self.head_proj(h)  # [B, T, K * 11]

        out_reshaped = out_flat.view(B, T, self.num_anchors, 11)
        conf_logits = out_reshaped[:, :, :, 0:1]
        confidences = torch.sigmoid(conf_logits)
        class_logits = out_reshaped[:, :, :, 1:4]
        box_offsets = out_reshaped[:, :, :, 4:11]

        # Apply offsets to anchor priors
        anchors_expanded = self.anchor_priors.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        # Center offsets + scale exp offsets
        pred_centers = anchors_expanded[:, :, :, :3] + box_offsets[:, :, :, :3]
        pred_dims = anchors_expanded[:, :, :, 3:6] * torch.exp(torch.clamp(box_offsets[:, :, :, 3:6], -1.0, 1.0))
        pred_yaw = anchors_expanded[:, :, :, 6:7] + box_offsets[:, :, :, 6:7]

        box_params = torch.cat([pred_centers, pred_dims, pred_yaw], dim=-1)

        if is_unbatched_seq:
            confidences = confidences.squeeze(1)
            class_logits = class_logits.squeeze(1)
            box_params = box_params.squeeze(1)

        return confidences, class_logits, box_params


class QueryBasedMultiObjectHead(nn.Module):
    """HEAD-2: Learnable Query-Based Multi-Object Prediction Head.

    Uses Q learned object query vectors that interact with temporal latents:
    For each query q in 1..Q:
    - Objectness confidence: p_q in [0, 1]
    - Class logits: c_q in R^3
    - Direct 3D box regression: [x, y, z, l, w, h, yaw]
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 64,
        num_queries: int = 16,
        num_classes: int = NUM_OBJECT_CLASSES,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Learnable Query Embeddings [Q, hidden_dim]
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # Cross-Interaction Block
        self.query_interaction = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.conf_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Linear(hidden_dim, 7)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass predicting multi-object candidates via query decoding.

        Returns:
            Tuple of (confidences [B, T, Q, 1], class_logits [B, T, Q, 3], box_params [B, T, Q, 7]).
        """
        is_unbatched_seq = z.dim() == 2
        if is_unbatched_seq:
            z = z.unsqueeze(1)

        B, T, D = z.shape
        # Expand queries: [B, T, Q, hidden_dim]
        q_exp = self.query_embed.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        # Expand latent z: [B, T, Q, D]
        z_exp = z.unsqueeze(2).expand(-1, -1, self.num_queries, -1)

        combined = torch.cat([z_exp, q_exp], dim=-1)  # [B, T, Q, D + hidden_dim]
        h = self.query_interaction(combined)          # [B, T, Q, hidden_dim]

        confidences = torch.sigmoid(self.conf_head(h)) # [B, T, Q, 1]
        class_logits = self.cls_head(h)               # [B, T, Q, 3]
        box_params = self.box_head(h)                 # [B, T, Q, 7]

        # Center in front: x in [0, 32], y in [-16, 16], z in [-2.5, 2.5]
        box_params_bounded = torch.cat([
            torch.clamp(box_params[:, :, :, 0:1], 0.0, 32.0),
            torch.clamp(box_params[:, :, :, 1:2], -16.0, 16.0),
            torch.clamp(box_params[:, :, :, 2:3], -2.5, 2.5),
            F.softplus(box_params[:, :, :, 3:6]) + 0.1,  # Positive dimensions (l, w, h)
            box_params[:, :, :, 6:7],                     # Yaw
        ], dim=-1)

        if is_unbatched_seq:
            confidences = confidences.squeeze(1)
            class_logits = class_logits.squeeze(1)
            box_params_bounded = box_params_bounded.squeeze(1)

        return confidences, class_logits, box_params_bounded
