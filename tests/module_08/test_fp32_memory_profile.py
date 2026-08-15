"""Unit test for FP32 model memory profile and parameter counts."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_05_latent_diffusion.denoiser import LightweightDenoiser
from module_05_latent_diffusion.scheduler import DDPMScheduler
from module_06_physics.latent_physics_head import LatentPhysicsHead


def test_fp32_parameter_counts_and_types():
    """Verify that all production model weights are strictly torch.float32 and parameter counts match."""
    encoder = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16, num_classes=4, use_attention=False)
    denoiser = LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2)
    physics_head = LatentPhysicsHead(latent_dim=64, hidden_dim=32)

    # Check dtypes
    for p in encoder.parameters():
        assert p.dtype == torch.float32, f"Encoder parameter {p} is not float32"
    for p in denoiser.parameters():
        assert p.dtype == torch.float32, f"Denoiser parameter {p} is not float32"
    for p in physics_head.parameters():
        assert p.dtype == torch.float32, f"Physics head parameter {p} is not float32"

    v0_params = sum(p.numel() for p in encoder.parameters())
    denoiser_params = sum(p.numel() for p in denoiser.parameters())
    physics_params = sum(p.numel() for p in physics_head.parameters())
    total_params = v0_params + denoiser_params + physics_params

    assert v0_params == 70566, f"Expected 70,566 PhotonV0 params, got {v0_params}"
    assert denoiser_params == 289344, f"Expected 289,344 Denoiser params, got {denoiser_params}"
    assert physics_params == 6339, f"Expected 6,339 Physics params, got {physics_params}"
    assert total_params == 366249, f"Expected 366,249 total params, got {total_params}"

    # Weight memory
    total_bytes = total_params * 4
    assert total_bytes == 1464996, f"Expected 1,464,996 bytes, got {total_bytes}"
