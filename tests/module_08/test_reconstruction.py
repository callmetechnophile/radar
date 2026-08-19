"""Unit tests for reconstruction head and composite loss functions."""

import pytest
import torch
from module_08_vod.reconstruction_head import OccupancyReconstructionHead
from module_08_vod.losses import OccupancyReconstructionLoss


def test_reconstruction_head_shapes():
    """Verify reconstruction head output shapes for 2D and 3D latents."""
    head = OccupancyReconstructionHead(in_dim=64, voxel_dims=(32, 32, 8))

    # Single frame [B, 64]
    z_single = torch.randn(4, 64)
    logits_single = head(z_single)
    assert logits_single.shape == (4, 32, 32, 8)

    # Sequence [B, T, 64]
    z_seq = torch.randn(2, 8, 64)
    logits_seq = head(z_seq)
    assert logits_seq.shape == (2, 8, 32, 32, 8)


def test_occupancy_loss_computation():
    """Verify occupancy loss calculation and gradient flow."""
    loss_fn = OccupancyReconstructionLoss(pos_weight=4.0, alpha=0.5)

    pred_logits = torch.randn(2, 8, 32, 32, 8, requires_grad=True)
    gt_occupancy = (torch.rand(2, 8, 32, 32, 8) > 0.9).float()
    mask = torch.ones(2, 8, 1)

    loss, comps = loss_fn(pred_logits, gt_occupancy, mask)
    assert loss.item() > 0
    assert "loss_bce" in comps
    assert "loss_dice" in comps

    loss.backward()
    assert pred_logits.grad is not None
    assert not torch.isnan(pred_logits.grad).any()
