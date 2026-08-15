"""Unit tests for Oxford Radar dataset loader."""

import pytest
import numpy as np

from module_07_temporal.oxford_adapter import OxfordRadarAdapter
from module_07_temporal.temporal_sequence import RadarFrame


def test_oxford_adapter_initialization():
    """Verify that OxfordRadarAdapter initializes cleanly and discovers all 51 radar scans."""
    adapter = OxfordRadarAdapter()
    assert adapter.num_scans == 51, f"Expected 51 scans in small sample, got {adapter.num_scans}"
    assert adapter.num_azimuths == 400
    assert adapter.num_range_bins == 3768
    assert adapter.range_resolution == pytest.approx(0.0432)
    assert adapter.max_range_m == pytest.approx(3768 * 0.0432)


def test_oxford_frame_loading():
    """Verify that single frames load with valid shapes, types, and values (no NaN/Inf)."""
    adapter = OxfordRadarAdapter()
    frame = adapter.load_frame(0)

    assert isinstance(frame, RadarFrame)
    assert frame.radar.shape == (400, 3768)
    assert frame.radar.dtype == np.float32
    assert frame.azimuths.shape == (400,)
    assert not np.isnan(frame.radar).any()
    assert not np.isinf(frame.radar).any()
    assert 0.0 <= np.min(frame.radar) <= 1.0
    assert 0.0 <= np.max(frame.radar) <= 1.0


def test_oxford_frame_indexing():
    """Verify that frame loading by index and by microsecond timestamp produce identical data."""
    adapter = OxfordRadarAdapter()
    ts0 = adapter.get_timestamp(0)
    frame_by_idx = adapter.load_frame(0)
    frame_by_ts = adapter.load_frame(ts0)

    assert np.array_equal(frame_by_idx.radar, frame_by_ts.radar)
    assert frame_by_idx.timestamp_us == frame_by_ts.timestamp_us
