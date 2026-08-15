"""Unit tests for Oxford radar timestamp utilities."""

import pytest
import numpy as np

from module_07_temporal.oxford_adapter import OxfordRadarAdapter
from module_07_temporal.timestamp_utils import compute_timestamp_statistics


def test_oxford_timestamps_monotonicity():
    """Verify that timestamps are strictly positive, monotonically increasing, and free of anomalies."""
    adapter = OxfordRadarAdapter()
    timestamps = adapter.get_timestamps()

    assert len(timestamps) == 51
    assert (timestamps > 0).all()
    assert (np.diff(timestamps) > 0).all(), "Timestamps are not strictly monotonically increasing!"


def test_oxford_temporal_statistics():
    """Verify calculated FPS and delta t characteristics of the Oxford Navtech radar."""
    adapter = OxfordRadarAdapter()
    timestamps = adapter.get_timestamps()
    stats = compute_timestamp_statistics(timestamps)

    assert stats["count"] == 51
    assert 3.5 <= stats["fps"] <= 4.5, f"Expected ~4.0 Hz FPS, got {stats['fps']}"
    assert 0.20 <= stats["dt_mean_s"] <= 0.30, f"Expected ~0.25s mean dt, got {stats['dt_mean_s']}"
    assert stats["dt_min_s"] > 0.10
    assert stats["dt_max_s"] < 0.50
    assert stats["jitter_s"] < 0.05, f"Temporal jitter too high: {stats['jitter_s']}"
    assert stats["total_duration_s"] > 10.0
