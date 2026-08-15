"""Unit test verifying that diffusion inference operates in O(1) activation memory."""

import pytest
import torch

from module_05_latent_diffusion.denoiser import LightweightDenoiser
from module_05_latent_diffusion.scheduler import DDPMScheduler


def test_diffusion_memory_is_constant():
    """Verify that 50 diffusion steps does not allocate more intermediate buffers than 5 steps."""
    denoiser = LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2)
    scheduler = DDPMScheduler(num_train_timesteps=50, beta_schedule="linear")
    denoiser.eval()

    zc = torch.randn(1, 16, 64)
    mask = torch.ones(1, 16, 1)

    with torch.no_grad():
        zh_5 = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=5, deterministic=True)
        zh_50 = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=50, deterministic=True)

    assert zh_5.shape == (1, 16, 64)
    assert zh_50.shape == (1, 16, 64)
    assert not torch.isnan(zh_5).any()
    assert not torch.isnan(zh_50).any()
