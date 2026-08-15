"""Unit tests for temporal reconstruction metrics and continuity error."""

import pytest
import numpy as np
import torch

from module_07_temporal.metrics import compute_reconstruction_metrics


def test_metrics_calculation_identical_inputs():
    """Verify that identical clean and predicted sequences yield exactly 0.0 error."""
    x = np.random.randn(4, 8, 64).astype(np.float32)
    mask = np.ones((4, 8, 1), dtype=np.float32)
    mask[:, 2:4] = 0.0  # missing

    metrics = compute_reconstruction_metrics(x, x, mask)

    assert metrics["missing_mse"] == 0.0
    assert metrics["missing_mae"] == 0.0
    assert metrics["missing_rmse"] == 0.0
    assert metrics["full_mse"] == 0.0
    assert metrics["temporal_error"] == 0.0


def test_temporal_continuity_error_positive_on_jitter():
    """Verify that artificial temporal discontinuity increases L_temporal."""
    x_clean = np.ones((1, 8, 10), dtype=np.float32)
    x_jitter = x_clean.copy()
    x_jitter[0, 3] += 5.0  # Spike at t=3

    mask = np.ones((1, 8, 1), dtype=np.float32)
    mask[0, 3] = 0.0

    metrics = compute_reconstruction_metrics(x_clean, x_jitter, mask)
    assert metrics["temporal_error"] > 0.0
    assert metrics["missing_mse"] == pytest.approx(25.0)
