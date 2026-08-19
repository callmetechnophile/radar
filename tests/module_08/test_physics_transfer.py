"""Unit tests for VoDPhysicsHead and VoDPhysicsLoss."""

import pytest
import torch
from module_08_vod.physics_head import VoDPhysicsHead, VoDPhysicsLoss


def test_physics_head_forward():
    """Verify kinematics prediction forward pass."""
    head = VoDPhysicsHead(in_dim=64, hidden_dim=32, num_outputs=5)
    z = torch.randn(2, 8, 64)
    kin = head(z)

    assert kin.shape == (2, 8, 5)
    assert not torch.isnan(kin).any()


def test_physics_loss_gradient():
    """Verify differentiable kinematic consistency loss."""
    loss_fn = VoDPhysicsLoss(dt=0.07692, lambda_disp=1.0, lambda_acc=0.1)
    kin = torch.randn(2, 8, 5, requires_grad=True)

    loss, comps = loss_fn(kin)
    assert loss.item() > 0
    assert "loss_displacement" in comps
    assert "loss_acceleration" in comps

    loss.backward()
    assert kin.grad is not None
    assert not torch.isnan(kin.grad).any()
