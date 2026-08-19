"""Unit tests for VoD Radar and LiDAR data loader and voxelization functions."""

from pathlib import Path
import numpy as np
import pytest
import torch

from module_08_vod.constants import RADAR_TRAIN_DIR, LIDAR_TRAIN_DIR, CALIB_RADAR_DIR, CALIB_LIDAR_DIR
from module_08_vod.radar_loader import (
    load_radar_point_cloud,
    load_lidar_point_cloud,
    load_calibration_txt,
    transform_lidar_to_radar,
    point_cloud_to_occupancy,
    occupancy_to_point_cloud,
)


def test_load_radar_point_cloud():
    """Verify loading single-scan radar point clouds with exact 7-field structure."""
    sample_file = RADAR_TRAIN_DIR / "00000.bin"
    if not sample_file.exists():
        pytest.skip("VoD dataset not available locally.")

    pts = load_radar_point_cloud(sample_file)
    assert isinstance(pts, np.ndarray)
    assert pts.ndim == 2
    assert pts.shape[1] == 7
    assert pts.dtype == np.float32
    assert len(pts) > 0
    assert not np.isnan(pts).any()
    assert not np.isinf(pts).any()


def test_load_lidar_point_cloud():
    """Verify loading LiDAR point clouds with 4-field structure."""
    sample_file = LIDAR_TRAIN_DIR / "00000.bin"
    if not sample_file.exists():
        pytest.skip("VoD dataset not available locally.")

    pts = load_lidar_point_cloud(sample_file)
    assert isinstance(pts, np.ndarray)
    assert pts.ndim == 2
    assert pts.shape[1] == 4
    assert pts.dtype == np.float32
    assert len(pts) > 0


def test_transform_lidar_to_radar():
    """Verify geometric transformation from LiDAR coordinates to Radar frame."""
    calib_rad_file = CALIB_RADAR_DIR / "00000.txt"
    calib_lid_file = CALIB_LIDAR_DIR / "00000.txt"
    lidar_file = LIDAR_TRAIN_DIR / "00000.bin"

    if not calib_rad_file.exists() or not lidar_file.exists():
        pytest.skip("VoD dataset not available locally.")

    cr = load_calibration_txt(calib_rad_file)
    cl = load_calibration_txt(calib_lid_file)
    pts_lid = load_lidar_point_cloud(lidar_file)

    pts_rad = transform_lidar_to_radar(pts_lid, cr, cl)
    assert pts_rad.shape == (len(pts_lid), 3)
    assert not np.isnan(pts_rad).any()
    # Check bounded coordinates
    assert np.all(pts_rad[:, 0] > -150.0) and np.all(pts_rad[:, 0] < 150.0)


def test_point_cloud_to_occupancy_and_back():
    """Verify deterministic conversion between 3D points and binary occupancy grid."""
    dummy_pts = np.array([
        [5.0, 0.0, 0.0],
        [10.0, 2.0, -1.0],
        [20.0, -4.0, 1.0],
    ], dtype=np.float32)

    occ = point_cloud_to_occupancy(dummy_pts, voxel_dims=(32, 32, 8))
    assert occ.shape == (32, 32, 8)
    assert occ.sum() == 3.0
    assert set(np.unique(occ)).issubset({0.0, 1.0})

    pts_recovered = occupancy_to_point_cloud(occ, threshold=0.5)
    assert len(pts_recovered) == 3
    assert pts_recovered.shape[1] == 3
