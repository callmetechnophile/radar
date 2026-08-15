"""Unit tests for deterministic temporal corruption on Oxford sequences."""

import pytest
import numpy as np

from module_07_temporal.temporal_corruption import TemporalRadarCorruption


def test_dropout_determinism_and_ratios():
    """Verify that TemporalRadarCorruption is bitwise deterministic and matches target drop ratios."""
    op1 = TemporalRadarCorruption(seed=42)
    op2 = TemporalRadarCorruption(seed=42)

    T = 16
    for p in [0.10, 0.20, 0.30, 0.40, 0.50]:
        mask1, stats1 = op1.apply_random_dropout(sequence_length=T, p_drop=p)
        mask2, stats2 = op2.apply_random_dropout(sequence_length=T, p_drop=p)

        assert np.array_equal(mask1, mask2), f"Corruption is non-deterministic at p={p}"
        assert mask1.shape == (T,)
        assert set(np.unique(mask1)).issubset({0.0, 1.0})
        assert stats1["sequence_length"] == T
        assert stats1["missing_frame_count"] == int(np.sum(mask1 == 0.0))


def test_contiguous_gap_corruption():
    """Verify contiguous gap masking and gap length bounds."""
    op = TemporalRadarCorruption(seed=100)
    T = 16

    for gap in [1, 2, 4, 8]:
        mask, stats = op.apply_contiguous_gap(sequence_length=T, gap_length=gap, start_idx=4)
        assert mask.shape == (T,)
        assert stats["missing_frame_count"] == gap
        assert (mask[4 : 4 + gap] == 0.0).all()
        assert stats["max_gap_length"] == gap
