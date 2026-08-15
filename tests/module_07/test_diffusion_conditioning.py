"""Unit tests for diffusion condition generation and time embeddings."""

import pytest
import torch

from module_07_temporal.latent_diffusion import (
    SinusoidalPosEmb,
    OxfordLatentDenoiser,
    OxfordDiffusionScheduler,
)


def test_sinusoidal_pos_embedding_shapes_and_values():
    """Verify positional embeddings produce correct finite dimensions without NaN/Inf."""
    emb = SinusoidalPosEmb(dim=64)
    t = torch.tensor([0, 10, 50, 99], dtype=torch.long)
    out = emb(t)

    assert out.shape == (4, 64)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_denoiser_conditioning_flow():
    """Verify denoiser accepts noisy inputs, diffusion steps, Mamba condition, and mask."""
    denoiser = OxfordLatentDenoiser(feature_dim=64, hidden_dim=128, time_dim=64, num_layers=2)
    denoiser.eval()

    B, T, D = 2, 16, 64
    x_noisy = torch.randn(B, T, D)
    t_diff = torch.tensor([5, 45], dtype=torch.long)
    condition = torch.randn(B, T, D)
    mask = torch.ones(B, T, 1)
    mask[:, 4:8] = 0.0

    with torch.no_grad():
        pred_noise = denoiser(x_noisy, t_diff, condition, mask)

    assert pred_noise.shape == (B, T, D)
    assert not torch.isnan(pred_noise).any()
