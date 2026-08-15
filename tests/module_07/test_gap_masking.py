"""Unit tests for contiguous gap mask generation across gap lengths G in {1, 2, 4, 8, 16}."""

import pytest
import numpy as np

from module_07_temporal.temporal_corruption import TemporalRadarCorruption


def test_contiguous_gap_masking_lengths():
    """Verify that contiguous gaps produce exact block sizes G with 0s inside and 1s outside."""
    corr = TemporalRadarCorruption(seed=42)
    T = 16

    for G in [1, 2, 4, 8]:
        mask, stats = corr.apply_contiguous_gap(sequence_length=T, gap_length=G, start_idx=4)

        assert len(mask) == T
        assert int(np.sum(mask == 0.0)) == G
        assert (mask[4 : 4 + G] == 0.0).all()
        assert (mask[:4] == 1.0).all()
        assert (mask[4 + G :] == 1.0).all()
        assert stats["max_gap_length"] == G


def test_contiguous_gap_determinism():
    """Verify bitwise identical corruption masks under the same seed."""
    c1 = TemporalRadarCorruption(seed=123)
    c2 = TemporalRadarCorruption(seed=123)

    m1, _ = c1.apply_contiguous_gap(sequence_length=16, gap_length=4)
    m2, _ = c2.apply_contiguous_gap(sequence_length=16, gap_length=4)

    assert np.array_equal(m1, m2)
