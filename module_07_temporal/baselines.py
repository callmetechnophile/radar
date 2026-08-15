"""Baseline reconstruction models for Oxford Radar temporal learning benchmark: B0 (Persistence) and B1 (Frame-wise)."""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistenceBaseline:
    """B0: Trivial Persistence Baseline.

    For missing frame at time t (where mask[t] == 0):
        x_hat(t) = x_observed(last_known_t)

    If all preceding frames are missing, backward-fills from first observed frame.
    Parameter count: 0.
    """

    def reconstruct(self, sequence: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Reconstruct corrupted sequence using forward-fill persistence.

        Args:
            sequence: Array of shape [T, D] (or [T, H, W]).
            mask: Binary mask of shape [T] where 1=observed, 0=missing.

        Returns:
            Reconstructed array of same shape [T, D].
        """
        reconstructed = sequence.copy()
        T = len(sequence)

        # Find first observed frame
        first_obs_idx = None
        for i in range(T):
            if mask[i] == 1.0:
                first_obs_idx = i
                break

        last_known = sequence[first_obs_idx].copy() if first_obs_idx is not None else np.zeros_like(sequence[0])

        for t in range(T):
            if mask[t] == 1.0:
                last_known = sequence[t].copy()
                reconstructed[t] = sequence[t]
            else:
                reconstructed[t] = last_known.copy()

        return reconstructed

    def reconstruct_torch(self, x_corr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """PyTorch batch implementation of persistence baseline [B, T, D]."""
        B, T, D = x_corr.shape
        out = x_corr.clone()

        for b in range(B):
            m_b = mask[b] if mask.dim() == 2 else mask[b, :, 0]
            last_known = torch.zeros(D, device=x_corr.device, dtype=x_corr.dtype)
            # Find first observed
            for t in range(T):
                if m_b[t] == 1.0:
                    last_known = x_corr[b, t].clone()
                    break

            for t in range(T):
                if m_b[t] == 1.0:
                    last_known = x_corr[b, t].clone()
                    out[b, t] = x_corr[b, t]
                else:
                    out[b, t] = last_known.clone()

        return out


class FramewiseBaseline(nn.Module):
    """B1: Small Frame-wise Reconstruction Autoencoder.

    Processes single frames independently without any temporal context.
    Input: single frame feature vector [B, D].
    Output: reconstructed frame feature vector [B, D].
    """

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on single frame [B, D] or applied frame-wise over sequence [B, T, D]."""
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
            out_flat = self.net(x_flat)
            return out_flat.view(B, T, D)
        return self.net(x)

    def reconstruct(self, x_corr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reconstruct unobserved frames using frame-wise model."""
        with torch.no_grad():
            preds = self.forward(x_corr)
            if mask.dim() == 2:
                m_exp = mask.unsqueeze(-1)
            else:
                m_exp = mask
            # Reconstructed: keep observed where m=1, use prediction where m=0
            out = x_corr * m_exp + preds * (1.0 - m_exp)
            return out
