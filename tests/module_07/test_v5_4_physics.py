"""Unit tests for Phase V5.4 Physics-Aware Mamba Temporal Model."""

import pytest
import torch
import numpy as np

from module_07_temporal.physics_mamba import OxfordPhysicsHead, OxfordPhysicsAwareMamba
from experiments.run_v5_4_physics_mamba import compute_physical_residuals


def test_physics_head_shapes_and_values():
    """Verify OxfordPhysicsHead produces 5-DoF kinematic outputs [B, T, 5]."""
    head = OxfordPhysicsHead(feature_dim=64, hidden_dim=32, num_outputs=5)
    head.eval()

    B, T, D = 3, 16, 64
    z = torch.randn(B, T, D)
    with torch.no_grad():
        out = head(z)

    assert out.shape == (B, T, 5)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_physics_mamba_forward_and_reconstruction():
    """Verify deterministic radar-only inference produces correct shapes and preserves observed frames."""
    model = OxfordPhysicsAwareMamba(feature_dim=64, hidden_dim=64, mamba_layers=2, physics_hidden_dim=32)
    model.eval()

    B, T, D = 2, 16, 64
    x_clean = torch.randn(B, T, D)
    mask = torch.ones(B, T, 1)
    mask[:, 4:10] = 0.0  # 6-frame gap
    x_corr = x_clean * mask

    with torch.no_grad():
        rec = model.reconstruct(x_corr, mask)

    assert rec.shape == (B, T, D)
    # Check observed regions are exactly preserved
    assert torch.allclose(rec[:, :4], x_clean[:, :4], atol=1e-6)
    assert torch.allclose(rec[:, 10:], x_clean[:, 10:], atol=1e-6)
    assert not torch.isnan(rec).any()


def test_physics_loss_gradient_flow():
    """Verify gradients backpropagate through both radar reconstruction and auxiliary physics head."""
    model = OxfordPhysicsAwareMamba(feature_dim=64, hidden_dim=64, mamba_layers=2, physics_hidden_dim=32)
    model.train()

    B, T, D = 2, 8, 64
    x_clean = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, 1)
    mask[:, 2:5] = 0.0
    kin_targets = torch.randn(B, T, 5)

    loss, loss_dict = model.compute_loss(x_clean, mask, kin_targets, lambda_phys=0.05)

    assert "loss_total" in loss_dict
    assert "loss_rec" in loss_dict
    assert "loss_phys" in loss_dict
    assert loss.item() > 0.0

    loss.backward()

    # Verify gradients exist and are finite on both Mamba layers and physics head
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(p.grad).any()
        assert not torch.isinf(p.grad).any()


def test_physical_residuals_calculation():
    """Verify compute_physical_residuals calculates correct metrics on missing slots."""
    B, T = 2, 8
    mask = torch.ones(B, T, 1)
    mask[:, 3:6] = 0.0

    gt_phys = torch.zeros(B, T, 5)
    gt_phys[:, :, 2] = 5.0  # vx = 5 m/s

    pred_phys = gt_phys.clone()
    pred_phys[:, :, 2] = 7.0  # vx = 7 m/s (error = 2 m/s)

    residuals = compute_physical_residuals(pred_phys, gt_phys, mask)

    assert abs(residuals["r_phys_vel"] - 2.0) < 1e-4
    assert residuals["r_motion"] == 0.0
