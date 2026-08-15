"""Unit tests for Oxford temporal sequence construction."""

import pytest
import numpy as np

from module_07_temporal.oxford_adapter import OxfordRadarAdapter
from module_07_temporal.temporal_sequence import RadarSequence
from module_07_temporal.timestamp_utils import find_temporal_windows


def test_sequence_construction():
    """Verify loading contiguous sequences for T=4, T=8, and T=16."""
    adapter = OxfordRadarAdapter()

    for T in [4, 8, 16]:
        seq = adapter.load_sequence(start_idx=0, sequence_length=T)
        assert isinstance(seq, RadarSequence)
        assert len(seq) == T
        assert len(seq.frames) == T
        assert seq.polar_tensor.shape == (T, 400, 3768)
        assert seq.timestamps_us.shape == (T,)
        assert 0.20 <= seq.dt <= 0.30
        assert seq.total_duration_s > 0.0


def test_temporal_sliding_windows():
    """Verify sliding temporal window calculation and rejected sequence counts."""
    adapter = OxfordRadarAdapter()
    timestamps = adapter.get_timestamps()

    for T in [4, 8, 16]:
        windows, rejected = find_temporal_windows(timestamps, window_length=T, max_allowed_gap_s=0.50)
        expected_windows = len(timestamps) - T + 1
        assert len(windows) == expected_windows, f"Expected {expected_windows} windows for T={T}, got {len(windows)}"
        assert rejected == 0, f"Expected 0 rejected windows, got {rejected}"
