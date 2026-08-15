"""Unit tests for Oxford Mamba Temporal Model architecture and gradient flow."""

import pytest
import torch
import torch.nn as nn

from module_07_temporal.mamba_temporal import OxfordMambaTemporalModel


def test_mamba_temporal_forward_and_shapes():
    """Verify forward pass output shape for various sequence lengths T in {4, 8, 16}."""
    model = OxfordMambaTemporalModel(feature_dim=64, hidden_dim=64, num_layers=2)
    model.eval()

    for T in [4, 8, 16]:
        x = torch.randn(2, T, 64)
        mask = torch.ones(2, T, 1)
        out = model(x, mask)

        assert out.shape == (2, T, 64)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


def test_mamba_temporal_gradient_flow():
    """Verify clean gradient propagation through all Mamba blocks."""
    model = OxfordMambaTemporalModel(feature_dim=64, hidden_dim=64, num_layers=2)
    model.train()

    x = torch.randn(4, 8, 64, requires_grad=True)
    mask = torch.ones(4, 8, 1)
    target = torch.randn(4, 8, 64)

    out = model(x, mask)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()

    assert x.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(param.grad).any()
