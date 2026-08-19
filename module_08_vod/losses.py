"""Differentiable 3D Occupancy Reconstruction Losses for VoD."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OccupancyReconstructionLoss(nn.Module):
    """Composite balanced binary occupancy loss with positive class weighting and soft Dice loss.

    Formula:
        L = alpha * L_pos_weighted_BCE + (1 - alpha) * L_Dice
    """

    def __init__(
        self,
        pos_weight: float = 4.0,
        alpha: float = 0.5,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.alpha = float(alpha)
        self.smooth = float(smooth)

    def forward(
        self,
        pred_logits: torch.Tensor,
        gt_occupancy: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute occupancy loss between predicted logits and ground truth binary occupancy.

        Args:
            pred_logits: Predicted occupancy logits `[B, T, Vx, Vy, Vz]` or `[B, Vx, Vy, Vz]`.
            gt_occupancy: Ground truth binary occupancy tensor `[B, T, Vx, Vy, Vz]` in {0, 1}.
            mask: Optional observation mask `[B, T, 1]`.

        Returns:
            Tuple of (total_loss, component_dict).
        """
        # Ensure dimensions match
        if pred_logits.shape != gt_occupancy.shape:
            raise ValueError(f"Shape mismatch: pred {pred_logits.shape} vs gt {gt_occupancy.shape}")

        pos_w = torch.tensor([self.pos_weight], device=pred_logits.device, dtype=pred_logits.dtype)
        bce_loss_raw = F.binary_cross_entropy_with_logits(
            pred_logits,
            gt_occupancy,
            pos_weight=pos_w,
            reduction="none",
        )

        probs = torch.sigmoid(pred_logits)

        # Apply temporal mask if provided
        if mask is not None:
            # Reshape mask to match spatial dimensions [B, T, 1, 1, 1]
            while mask.dim() < pred_logits.dim():
                mask = mask.unsqueeze(-1)
            bce_loss = torch.sum(bce_loss_raw * mask) / torch.sum(mask).clamp(min=1e-6)

            # Masked Soft Dice Loss
            probs_m = probs * mask
            gt_m = gt_occupancy * mask
            intersection = torch.sum(probs_m * gt_m)
            cardinality = torch.sum(probs_m + gt_m)
            dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        else:
            bce_loss = torch.mean(bce_loss_raw)
            intersection = torch.sum(probs * gt_occupancy)
            cardinality = torch.sum(probs + gt_occupancy)
            dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        total_loss = self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss

        components = {
            "loss_total": total_loss,
            "loss_bce": bce_loss,
            "loss_dice": dice_loss,
        }
        return total_loss, components
