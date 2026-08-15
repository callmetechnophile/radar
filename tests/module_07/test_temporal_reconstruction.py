"""Unit tests for temporal sequence reconstruction and inpainting preservation."""

import pytest
import torch
import numpy as np

from module_07_temporal.baselines import PersistenceBaseline, FramewiseBaseline
from module_07_temporal.mamba_temporal import OxfordMambaTemporalModel


def test_observed_frame_preservation():
    """Verify that reconstruct() methods preserve observed frames exactly where mask == 1."""
    x = torch.randn(2, 8, 64)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                         [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]).unsqueeze(-1)
    x_corr = x * mask

    b0 = PersistenceBaseline()
    b1 = FramewiseBaseline(64, 128)
    b2 = OxfordMambaTemporalModel(64, 64, 2)

    rec_b0 = b0.reconstruct_torch(x_corr, mask)
    rec_b1 = b1.reconstruct(x_corr, mask)
    rec_b2 = b2.reconstruct(x_corr, mask)

    # Where mask == 1, reconstructed output must match original x exactly
    for b in range(2):
        for t in range(8):
            if mask[b, t, 0] == 1.0:
                assert torch.allclose(rec_b0[b, t], x[b, t], atol=1e-6)
                assert torch.allclose(rec_b1[b, t], x[b, t], atol=1e-6)
                assert torch.allclose(rec_b2[b, t], x[b, t], atol=1e-6)
