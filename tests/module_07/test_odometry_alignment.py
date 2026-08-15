"""Unit tests for Oxford radar odometry synchronization."""

import pytest
import numpy as np

from module_07_temporal.oxford_adapter import OxfordRadarAdapter


def test_odometry_pose_retrieval_and_synchronization():
    """Verify that 6-DoF odometry is synchronized with radar scans with low temporal error."""
    adapter = OxfordRadarAdapter()
    assert adapter.odometry_data is not None, "Odometry data must be present in sample dataset"

    # Query first 10 scans
    for idx in range(10):
        odom = adapter.get_odometry(idx)
        assert odom is not None
        assert "pose" in odom
        assert len(odom["pose"]) == 6
        assert odom["sync_error_s"] < 0.20, f"Synchronization error too high: {odom['sync_error_s']}s"
        assert not np.isnan(odom["pose"]).any()


def test_sequence_odometry_attachment():
    """Verify that loaded sequences contain synchronized pose matrices [T, 6]."""
    adapter = OxfordRadarAdapter()
    seq = adapter.load_sequence(start_idx=0, sequence_length=8)

    assert seq.odometry_poses is not None
    assert seq.odometry_poses.shape == (8, 6)
    assert not np.isnan(seq.odometry_poses).any()
