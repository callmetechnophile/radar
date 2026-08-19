"""Unit tests for VoD data integrity, finite values, and absence of NaN/Inf."""

import os
from pathlib import Path
import numpy as np
import pytest

from module_08_vod.constants import RADAR_TRAIN_DIR, LIDAR_TRAIN_DIR
from module_08_vod.radar_loader import load_radar_point_cloud, load_lidar_point_cloud


def test_radar_and_lidar_integrity_multiple_frames():
    """Verify finite values, no NaN/Inf across first 10 frames."""
    if not RADAR_TRAIN_DIR.exists() or not LIDAR_TRAIN_DIR.exists():
        pytest.skip("VoD dataset not available.")

    for i in range(10):
        rf = RADAR_TRAIN_DIR / f"{i:05d}.bin"
        lf = LIDAR_TRAIN_DIR / f"{i:05d}.bin"
        if rf.exists():
            rpts = load_radar_point_cloud(rf)
            assert not np.isnan(rpts).any()
            assert not np.isinf(rpts).any()
            assert rpts.shape[1] == 7

        if lf.exists():
            lpts = load_lidar_point_cloud(lf)
            assert not np.isnan(lpts).any()
            assert not np.isinf(lpts).any()
            assert lpts.shape[1] == 4
