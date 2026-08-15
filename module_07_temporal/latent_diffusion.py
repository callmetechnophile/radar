"""V5-specific Conditional Latent Diffusion Model for Oxford Radar Long-Contiguous-Gap Inpainting."""

from __future__ import annotations

import math
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_07_temporal.mamba_temporal import OxfordMambaTemporalModel


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class OxfordLatentDenoiser(nn.Module):
    """Conditional Denoising Network for Oxford Radar Latent Sequences [B, T, D].

    Predicts residual noise epsilon conditioned on:
    1. Noisy residual latent x_k [B, T, D]
    2. Diffusion timestep t_diff [B]
    3. Mamba temporal contextual condition c [B, T, D]
    4. Observation mask m [B, T, 1]
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Timestep MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: noisy residual (D) + condition (D) + mask (1) -> hidden_dim
        in_channels = feature_dim * 2 + 1
        self.in_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Denoising ResNet blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Output projection -> predicted noise [B, T, D]
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_diff: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass predicting added noise."""
        B, T, D = x_noisy.shape

        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        # Compute time embedding
        t_emb = self.time_mlp(t_diff)  # [B, hidden_dim]
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]

        # Concatenate noisy residual, condition, and mask
        in_feat = torch.cat([x_noisy, condition, mask], dim=-1)  # [B, T, 2*D + 1]
        h = self.in_proj(in_feat) + t_emb_expanded

        for block in self.blocks:
            h = h + block(h)

        pred_noise = self.out_proj(h)
        return pred_noise


class OxfordDiffusionScheduler:
    """Gaussian Diffusion Scheduler for training and fast DDIM/DDPM inference."""

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device: torch.device) -> OxfordDiffusionScheduler:
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise


class OxfordMambaLatentDiffusion(nn.Module):
    """B3: Residual Mamba + Latent Diffusion Inpainting System."""

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        mamba_layers: int = 2,
        denoiser_layers: int = 3,
        num_train_timesteps: int = 100,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 1. Mamba Temporal Prior
        self.mamba_encoder = OxfordMambaTemporalModel(
            feature_dim=feature_dim,
            hidden_dim=feature_dim,
            num_layers=mamba_layers,
        )

        # 2. Residual Denoiser
        self.denoiser = OxfordLatentDenoiser(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            time_dim=64,
            num_layers=denoiser_layers,
        )

        # 3. Diffusion Scheduler
        self.scheduler = OxfordDiffusionScheduler(
            num_train_timesteps=num_train_timesteps,
            device=device,
        )

    def compute_condition(self, x_corr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mamba_encoder(x_corr, mask)

    def forward_loss(
        self,
        x_clean: torch.Tensor,
        mask: torch.Tensor,
        lambda_rec: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss on residual target."""
        B, T, D = x_clean.shape
        device = x_clean.device

        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        x_corr = x_clean * mask
        mamba_pred = self.compute_condition(x_corr, mask)

        # Target is the high-frequency residual on missing slots
        residual_target = (x_clean - mamba_pred) * (1.0 - mask)

        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (B,), device=device, dtype=torch.long
        )
        noise = torch.randn_like(x_clean) * (1.0 - mask)
        noisy_residual = self.scheduler.add_noise(residual_target, noise, timesteps) * (1.0 - mask)

        pred_noise = self.denoiser(noisy_residual, timesteps, mamba_pred, mask)

        diff_loss = F.mse_loss(pred_noise * (1.0 - mask), noise * (1.0 - mask))
        recon_loss = F.mse_loss(mamba_pred * (1.0 - mask), x_clean * (1.0 - mask))

        total_loss = diff_loss + lambda_rec * recon_loss

        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_diffusion": float(diff_loss.item()),
            "loss_recon": float(recon_loss.item()),
        }

    @torch.no_grad()
    def sample(
        self,
        x_corr: torch.Tensor,
        mask: torch.Tensor,
        num_inference_steps: int = 10,
    ) -> torch.Tensor:
        """Fast DDIM-style residual denoising."""
        B, T, D = x_corr.shape
        device = x_corr.device

        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        mamba_pred = self.compute_condition(x_corr, mask)

        step_stride = max(1, self.scheduler.num_train_timesteps // num_inference_steps)
        inference_timesteps = list(range(0, self.scheduler.num_train_timesteps, step_stride))[::-1]
        inference_timesteps = inference_timesteps[:num_inference_steps]

        # Start from small noise for residual
        res_t = torch.randn_like(x_corr) * 0.1 * (1.0 - mask)

        for i, t_idx in enumerate(inference_timesteps):
            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
            pred_noise = self.denoiser(res_t, t_batch, mamba_pred, mask) * (1.0 - mask)

            alpha_bar_t = self.scheduler.alphas_cumprod[t_idx]
            prev_t_idx = inference_timesteps[i + 1] if i + 1 < len(inference_timesteps) else 0
            alpha_bar_prev = (
                self.scheduler.alphas_cumprod[prev_t_idx]
                if prev_t_idx > 0
                else torch.tensor(1.0, device=device)
            )

            pred_res0 = (res_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_res0 = torch.clamp(pred_res0, -2.0, 2.0)

            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=0.0)) * pred_noise
            res_prev = torch.sqrt(alpha_bar_prev) * pred_res0 + dir_xt
            res_t = res_prev * (1.0 - mask)

        # Final reconstructed output: Observed + Mamba prior + refined residual
        pred_clean = (mamba_pred + res_t) * (1.0 - mask)
        out = x_corr * mask + pred_clean
        return out
