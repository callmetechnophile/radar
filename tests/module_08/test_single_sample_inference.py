"""Unit test for complete single-sample FP32 pipeline integration."""

import pytest
import torch
import torch.nn.functional as F

from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_05_latent_diffusion.denoiser import LightweightDenoiser
from module_05_latent_diffusion.scheduler import DDPMScheduler
from module_06_physics.radar_constants import DT
from module_06_physics.latent_physics_head import LatentPhysicsHead
from module_07_adaptive_compute import AdaptiveComputeStateEncoder, RuleBasedDiffusionScheduler


def test_complete_fp32_pipeline_b1():
    """Verify end-to-end forward execution of complete FP32 system at B=1."""
    encoder = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16, num_classes=4, use_attention=False)
    denoiser = LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2)
    scheduler = DDPMScheduler(num_train_timesteps=50, beta_schedule="linear")
    physics_head = LatentPhysicsHead(latent_dim=64, hidden_dim=32)
    state_encoder = AdaptiveComputeStateEncoder(physics_head=physics_head, dt=DT)
    rule_scheduler = RuleBasedDiffusionScheduler()

    encoder.eval()
    denoiser.eval()
    physics_head.eval()

    x = torch.randn(1, 16, 64)
    mask = torch.ones(1, 16, 1)
    mask[:, 8:, :] = 0.0

    with torch.no_grad():
        # 1. Mamba extract
        z0, _ = encoder.extract_latents(x)
        zc = z0 * mask

        # 2. State & Rule action
        s_vec, s_dict = state_encoder(zc, mask)
        action_steps = rule_scheduler.predict_action(s_vec[0])

        assert action_steps in [5, 10, 20, 50]
        assert s_vec.shape == (1, 9)

        # 3. Diffusion inpaint
        zh = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=action_steps, deterministic=True)
        assert zh.shape == (1, 16, 64)

        # 4. Perception & Physics
        logits = encoder.classification_head(zh[:, -1, :])
        obs = physics_head(zh)

        assert logits.shape == (1, 4)
        assert obs["range"].shape == (1, 16)
        assert obs["velocity"].shape == (1, 16)
        assert obs["energy"].shape == (1, 16)
