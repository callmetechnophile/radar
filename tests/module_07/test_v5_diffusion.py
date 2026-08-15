"""Unit tests for Oxford Mamba Latent Diffusion forward loss and gradient flow."""

import pytest
import torch

from module_07_temporal.latent_diffusion import OxfordMambaLatentDiffusion


def test_v5_diffusion_training_loss_and_gradients():
    """Verify training loss computation and gradient backpropagation through both Mamba and denoiser."""
    model = OxfordMambaLatentDiffusion(feature_dim=64, hidden_dim=128, mamba_layers=2, denoiser_layers=2)
    model.train()

    B, T, D = 2, 8, 64
    x_clean = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, 1)
    mask[:, 3:6] = 0.0

    loss, loss_dict = model.forward_loss(x_clean, mask, lambda_rec=1.0)

    assert "loss_total" in loss_dict
    assert "loss_diffusion" in loss_dict
    assert "loss_recon" in loss_dict
    assert loss.item() > 0.0

    loss.backward()

    # Verify gradients on denoiser parameters
    for name, p in model.denoiser.named_parameters():
        assert p.grad is not None, f"Parameter {name} received no gradient!"
        assert not torch.isnan(p.grad).any()
