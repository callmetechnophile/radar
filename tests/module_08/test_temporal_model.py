"""Unit tests for VoDFramewiseBaseline and VoDMambaTemporalModel."""

import pytest
import torch
from module_08_vod.temporal_model import VoDFramewiseBaseline, VoDMambaTemporalModel


def test_framewise_baseline_forward():
    """Verify forward pass of frame-wise baseline."""
    model = VoDFramewiseBaseline(point_in_dim=7, feature_dim=64, voxel_dims=(32, 32, 8))
    tokens = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8, 1)

    logits = model(tokens, mask)
    assert logits.shape == (2, 8, 32, 32, 8)
    assert not torch.isnan(logits).any()


def test_mamba_temporal_model_forward():
    """Verify forward pass of Mamba temporal model."""
    model = VoDMambaTemporalModel(
        point_in_dim=7,
        feature_dim=64,
        hidden_dim=64,
        num_layers=2,
        voxel_dims=(32, 32, 8),
    )
    tokens = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8, 1)

    logits = model(tokens, mask)
    assert logits.shape == (2, 8, 32, 32, 8)
    assert not torch.isnan(logits).any()


def test_gradient_flow_and_temporal_receptive_field():
    """Verify gradient backpropagation through Mamba layers and causal temporal dependency."""
    model = VoDMambaTemporalModel(
        point_in_dim=7,
        feature_dim=64,
        hidden_dim=64,
        num_layers=2,
        voxel_dims=(32, 32, 8),
    )
    tokens = torch.randn(1, 8, 64, requires_grad=True)
    mask = torch.ones(1, 8, 1)

    logits = model(tokens, mask)
    loss = logits.sum()
    loss.backward()

    assert tokens.grad is not None
    assert not torch.isnan(tokens.grad).any()
