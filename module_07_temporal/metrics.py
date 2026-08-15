"""Evaluation metrics for Oxford Radar temporal reconstruction benchmark."""

from __future__ import annotations

from typing import Dict, Any, Union
import numpy as np
import torch


def compute_reconstruction_metrics(
    x_clean: Union[np.ndarray, torch.Tensor],
    x_hat: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """Compute comprehensive reconstruction and temporal continuity metrics.

    Args:
        x_clean: Ground truth sequence [B, T, D] or [T, D].
        x_hat: Reconstructed sequence [B, T, D] or [T, D].
        mask: Binary mask where 1=observed, 0=missing [B, T, 1] or [B, T] or [T].

    Returns:
        Dictionary containing missing MSE, MAE, RMSE, full MSE, and temporal continuity error.
    """
    if isinstance(x_clean, torch.Tensor):
        x_clean = x_clean.detach().cpu().numpy()
    if isinstance(x_hat, torch.Tensor):
        x_hat = x_hat.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if x_clean.ndim == 2:
        x_clean = np.expand_dims(x_clean, 0)
        x_hat = np.expand_dims(x_hat, 0)
        mask = np.expand_dims(mask, 0)

    if mask.ndim == 2:
        mask = np.expand_dims(mask, -1)  # [B, T, 1]

    missing_mask = (mask == 0.0)  # [B, T, 1]
    missing_mask_expanded = np.broadcast_to(missing_mask, x_clean.shape)

    # 1. Missing-frame Metrics
    if np.any(missing_mask):
        diff_missing = (x_hat - x_clean)[missing_mask_expanded]
        missing_mse = float(np.mean(diff_missing**2))
        missing_mae = float(np.mean(np.abs(diff_missing)))
        missing_rmse = float(np.sqrt(missing_mse))
    else:
        missing_mse = 0.0
        missing_mae = 0.0
        missing_rmse = 0.0

    # 2. Full-sequence MSE
    full_mse = float(np.mean((x_hat - x_clean)**2))

    # 3. Temporal Continuity Error: L_temporal = mean(||Delta x_hat - Delta x_clean||)
    # Inter-frame difference along time dimension (axis 1)
    if x_clean.shape[1] > 1:
        delta_clean = x_clean[:, 1:] - x_clean[:, :-1]
        delta_hat = x_hat[:, 1:] - x_hat[:, :-1]
        temporal_error = float(np.mean(np.abs(delta_hat - delta_clean)))
    else:
        temporal_error = 0.0

    return {
        "missing_mse": missing_mse,
        "missing_mae": missing_mae,
        "missing_rmse": missing_rmse,
        "full_mse": full_mse,
        "temporal_error": temporal_error,
    }
