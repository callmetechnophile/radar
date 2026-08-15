"""Unit test for FP32 single-sample deterministic inference."""

import pytest
import torch
import torch.nn.functional as F

from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_05_latent_diffusion.denoiser import LightweightDenoiser
from module_05_latent_diffusion.scheduler import DDPMScheduler
from module_06_physics.latent_physics_head import LatentPhysicsHead


def test_fp32_inference_determinism():
    """Verify that two sequential forward passes with identical inputs produce bitwise identical outputs."""
    encoder = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16, num_classes=4, use_attention=False)
    denoiser = LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2)
    scheduler = DDPMScheduler(num_train_timesteps=50, beta_schedule="linear")
    physics_head = LatentPhysicsHead(latent_dim=64, hidden_dim=32)

    encoder.eval()
    denoiser.eval()
    physics_head.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 16, 64)
    mask = torch.ones(1, 16, 1)
    mask[:, 5:10, :] = 0.0

    with torch.no_grad():
        z0_1, _ = encoder.extract_latents(x)
        zh_1 = scheduler.reconstruct(denoiser, z0_1 * mask, mask, num_inference_steps=10, deterministic=True)
        logits_1 = encoder.classification_head(zh_1[:, -1, :])
        obs_1 = physics_head(zh_1)

        z0_2, _ = encoder.extract_latents(x)
        zh_2 = scheduler.reconstruct(denoiser, z0_2 * mask, mask, num_inference_steps=10, deterministic=True)
        logits_2 = encoder.classification_head(zh_2[:, -1, :])
        obs_2 = physics_head(zh_2)

    assert torch.equal(z0_1, z0_2), "Extracted latents are not bitwise identical!"
    assert torch.equal(zh_1, zh_2), "Reconstructed latents are not bitwise identical!"
    assert torch.equal(logits_1, logits_2), "Classification logits are not bitwise identical!"
    assert torch.equal(obs_1["range"], obs_2["range"]), "Physics range outputs are not bitwise identical!"
