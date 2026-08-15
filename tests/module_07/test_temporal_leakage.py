"""Unit tests for temporal causality and zero future-frame leakage in Oxford temporal models."""

import pytest
import torch

from module_07_temporal.mamba_temporal import OxfordMambaTemporalModel
from module_07_temporal.baselines import PersistenceBaseline, FramewiseBaseline


def test_mamba_causality_and_no_future_leakage():
    """Verify that modifying future frame inputs x[:, t_future] does NOT alter predictions at t_current < t_future."""
    model = OxfordMambaTemporalModel(feature_dim=64, hidden_dim=64, num_layers=2)
    model.eval()

    T = 8
    x_base = torch.randn(1, T, 64)
    mask = torch.ones(1, T, 1)

    with torch.no_grad():
        out_base = model(x_base, mask)

    # Modify future frames at t >= 5
    x_perturbed = x_base.clone()
    x_perturbed[:, 5:] = torch.randn(1, T - 5, 64) * 10.0

    with torch.no_grad():
        out_perturbed = model(x_perturbed, mask)

    # Prediction at t=0 should be identical
    diff_t0 = torch.abs(out_base[:, 0] - out_perturbed[:, 0]).max().item()
    assert diff_t0 == pytest.approx(0.0, abs=1e-5), f"Future frame modification leaked to t=0! (diff={diff_t0})"


def test_persistence_causality():
    """Verify that Persistence baseline only propagates information forward."""
    b0 = PersistenceBaseline()
    T = 6
    x = torch.arange(T, dtype=torch.float32).unsqueeze(-1).expand(-1, 4).unsqueeze(0)  # [1, 6, 4]
    mask = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])  # missing at t=1, 2, 4

    rec = b0.reconstruct_torch(x * mask.unsqueeze(-1), mask.unsqueeze(-1))

    # Frame 1 and 2 must equal Frame 0
    assert torch.equal(rec[0, 1], x[0, 0])
    assert torch.equal(rec[0, 2], x[0, 0])
    # Frame 4 must equal Frame 3
    assert torch.equal(rec[0, 4], x[0, 3])
