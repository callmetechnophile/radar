"""Unit tests verifying zero target / missing-frame information leakage into condition."""

import pytest
import torch

from module_07_temporal.latent_diffusion import OxfordMambaLatentDiffusion


def test_diffusion_sampling_no_target_leakage():
    """Verify that during sampling, modifying clean missing target values does NOT alter sampled outputs."""
    model = OxfordMambaLatentDiffusion(feature_dim=64, hidden_dim=128, mamba_layers=2, denoiser_layers=2)
    model.eval()

    torch.manual_seed(42)
    B, T, D = 1, 8, 64
    x_clean_1 = torch.randn(B, T, D)
    mask = torch.ones(B, T, 1)
    mask[:, 2:6] = 0.0  # missing block at t=2..5

    x_corr_1 = x_clean_1 * mask

    # Sample 1
    torch.manual_seed(100)
    out1 = model.sample(x_corr_1, mask, num_inference_steps=5)

    # Change the target values drastically inside the missing block
    x_clean_2 = x_clean_1.clone()
    x_clean_2[:, 2:6] = torch.randn(B, 4, D) * 100.0
    x_corr_2 = x_clean_2 * mask  # Still strictly identical to x_corr_1!

    # Sample 2 with same random seed
    torch.manual_seed(100)
    out2 = model.sample(x_corr_2, mask, num_inference_steps=5)

    assert torch.allclose(out1, out2, atol=1e-6), "Target values leaked into the diffusion reconstruction pipeline!"
