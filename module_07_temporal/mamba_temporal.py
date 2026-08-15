"""B2: Mamba Temporal Model for Oxford Radar Temporal Sequence Inpainting and Reconstruction."""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_04_mamba_hybrid.mamba_core import MiniMambaBlock


class OxfordMambaTemporalModel(nn.Module):
    """B2: Causal Mamba Selective SSM Temporal Reconstruction Network.

    Processes corrupted temporal sequences of Oxford radar features [B, T, D]
    with observation masks [B, T, 1] using state-space temporal modeling.

    Architecture:
        Input [B, T, D + 1] (Feature vector concatenated with observation mask)
            ↓
        Linear Projection (D+1 -> hidden_dim) + LayerNorm
            ↓
        Mamba Block 1 (Selective SSM + Causal Conv1D + SiLU + Residual)
            ↓
        Mamba Block 2 (Selective SSM + Causal Conv1D + SiLU + Residual)
            ↓
        Final LayerNorm
            ↓
        Reconstruction Head (hidden_dim -> D)
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. Temporal Input Projection (Features + Mask channel)
        self.in_proj = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 2. Sequential Mamba SSM Blocks
        self.layers = nn.ModuleList([
            MiniMambaBlock(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # 3. Temporal Reconstruction Head
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x_corr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting clean temporal feature sequence.

        Args:
            x_corr: Corrupted sequence tensor of shape [B, T, feature_dim].
            mask: Binary observation mask of shape [B, T, 1] (or [B, T]).

        Returns:
            Reconstructed feature sequence of shape [B, T, feature_dim].
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        # Concatenate features and mask
        in_tensor = torch.cat([x_corr, mask], dim=-1)  # [B, T, D + 1]
        h = self.in_proj(in_tensor)                     # [B, T, hidden_dim]

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        delta_x = self.recon_head(h)                    # [B, T, feature_dim]
        pred_x = x_corr + delta_x                       # Residual prediction on top of input
        return pred_x

    def reconstruct(self, x_corr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Inpaint missing frames while preserving observed frames."""
        with torch.no_grad():
            preds = self.forward(x_corr, mask)
            if mask.dim() == 2:
                m_exp = mask.unsqueeze(-1)
            else:
                m_exp = mask
            out = x_corr * m_exp + preds * (1.0 - m_exp)
            return out
