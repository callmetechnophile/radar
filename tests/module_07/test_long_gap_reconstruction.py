"""Unit tests for long contiguous gap inpainting preservation and determinism."""

import pytest
import torch

from module_07_temporal.latent_diffusion import OxfordMambaLatentDiffusion


def test_sampling_preserves_observed_regions():
    """Verify that reverse diffusion sampling exactly retains clean values at observed indices (mask == 1)."""
    model = OxfordMambaLatentDiffusion(feature_dim=64, hidden_dim=128, mamba_layers=2, denoiser_layers=2)
    model.eval()

    B, T, D = 2, 16, 64
    x_clean = torch.randn(B, T, D)
    mask = torch.ones(B, T, 1)
    mask[:, 4:12] = 0.0  # 8-frame gap
    x_corr = x_clean * mask

    out = model.sample(x_corr, mask, num_inference_steps=5)

    assert out.shape == (B, T, D)
    # Check observed region exactly preserved
    assert torch.allclose(out[:, :4], x_clean[:, :4], atol=1e-6)
    assert torch.allclose(out[:, 12:], x_clean[:, 12:], atol=1e-6)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
