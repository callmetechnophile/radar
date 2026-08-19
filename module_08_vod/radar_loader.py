"""Native Radar Point Cloud Loader, Calibration Transforms, and 3D Voxelization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch

from module_08_vod.constants import (
    RADAR_POINT_CHANNELS,
    VOXEL_X_MIN,
    VOXEL_X_MAX,
    VOXEL_Y_MIN,
    VOXEL_Y_MAX,
    VOXEL_Z_MIN,
    VOXEL_Z_MAX,
    VOXEL_DIM_X,
    VOXEL_DIM_Y,
    VOXEL_DIM_Z,
)


def load_radar_point_cloud(bin_path: Union[str, Path]) -> np.ndarray:
    """Load native single-scan VoD radar point cloud from a .bin file.

    Args:
        bin_path: Path to the radar .bin file.

    Returns:
        Array of shape [N, 7] and type float32 containing [x, y, z, rcs, v_r, v_r_comp, time_id].

    Raises:
        ValueError: If array shape is invalid or contains NaN/Inf values.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(bin_path)
    if not path.exists():
        raise FileNotFoundError(f"Radar file not found: {path}")

    raw = np.fromfile(str(path), dtype=np.float32)
    if len(raw) % RADAR_POINT_CHANNELS != 0:
        raise ValueError(
            f"Corrupted radar file {path.name}: raw length {len(raw)} is not divisible by {RADAR_POINT_CHANNELS}"
        )

    pts = raw.reshape(-1, RADAR_POINT_CHANNELS)

    if np.isnan(pts).any():
        raise ValueError(f"Radar point cloud {path.name} contains NaN values!")
    if np.isinf(pts).any():
        raise ValueError(f"Radar point cloud {path.name} contains Inf values!")

    return pts


def load_lidar_point_cloud(bin_path: Union[str, Path]) -> np.ndarray:
    """Load native VoD LiDAR point cloud from a .bin file.

    Args:
        bin_path: Path to the lidar .bin file.

    Returns:
        Array of shape [M, 4] and type float32 containing [x, y, z, reflectance].
    """
    path = Path(bin_path)
    if not path.exists():
        raise FileNotFoundError(f"LiDAR file not found: {path}")

    raw = np.fromfile(str(path), dtype=np.float32)
    if len(raw) % 4 != 0:
        raise ValueError(f"Corrupted LiDAR file {path.name}: raw length {len(raw)} not divisible by 4")

    pts = raw.reshape(-1, 4)
    return pts


def load_calibration_txt(calib_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Parse KITTI-format calibration text file into numpy matrices.

    Args:
        calib_path: Path to calibration .txt file.

    Returns:
        Dictionary of calibration matrices (P0, P1, P2, P3, R0_rect, Tr_velo_to_cam).
    """
    path = Path(calib_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    matrices = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                vals = [float(x) for x in val.strip().split() if x]
                matrices[key.strip()] = np.array(vals, dtype=np.float64)
    return matrices


def transform_lidar_to_radar(
    pts_lidar: np.ndarray,
    calib_radar: Dict[str, np.ndarray],
    calib_lidar: Dict[str, np.ndarray],
) -> np.ndarray:
    """Transform LiDAR 3D coordinates into the Radar coordinate frame using extrinsics.

    P_cam = R_lid * P_lid + t_lid
    P_rad = R_rad^(-1) * (P_cam - t_rad)
    """
    Tr_rad = calib_radar["Tr_velo_to_cam"].reshape(3, 4)
    Tr_lid = calib_lidar["Tr_velo_to_cam"].reshape(3, 4)

    R_rad, t_rad = Tr_rad[:, :3], Tr_rad[:, 3]
    R_lid, t_lid = Tr_lid[:, :3], Tr_lid[:, 3]

    xyz_lid = pts_lidar[:, :3]
    # Transform LiDAR to Camera frame
    xyz_cam = np.dot(R_lid, xyz_lid.T).T + t_lid

    # Transform Camera to Radar frame
    R_rad_inv = np.linalg.inv(R_rad)
    xyz_rad = np.dot(R_rad_inv, (xyz_cam - t_rad).T).T

    return xyz_rad.astype(np.float32)


def point_cloud_to_occupancy(
    pts_xyz: np.ndarray,
    x_range: Tuple[float, float] = (VOXEL_X_MIN, VOXEL_X_MAX),
    y_range: Tuple[float, float] = (VOXEL_Y_MIN, VOXEL_Y_MAX),
    z_range: Tuple[float, float] = (VOXEL_Z_MIN, VOXEL_Z_MAX),
    voxel_dims: Tuple[int, int, int] = (VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z),
) -> np.ndarray:
    """Convert 3D point cloud into a deterministic binary occupancy voxel grid.

    Args:
        pts_xyz: [N, 3] points in radar coordinate frame.
        x_range: (x_min, x_max) in meters.
        y_range: (y_min, y_max) in meters.
        z_range: (z_min, z_max) in meters.
        voxel_dims: (Vx, Vy, Vz) number of bins.

    Returns:
        Occupancy grid array of shape (Vx, Vy, Vz) with values {0.0, 1.0}.
    """
    vx, vy, vz = voxel_dims
    occ = np.zeros((vx, vy, vz), dtype=np.float32)

    if len(pts_xyz) == 0:
        return occ

    x, y, z = pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2]

    # Filter bounding box
    mask = (
        (x >= x_range[0])
        & (x < x_range[1])
        & (y >= y_range[0])
        & (y < y_range[1])
        & (z >= z_range[0])
        & (z < z_range[1])
    )

    if not mask.any():
        return occ

    x_valid, y_valid, z_valid = x[mask], y[mask], z[mask]

    # Map to integer indices
    ix = np.floor((x_valid - x_range[0]) / (x_range[1] - x_range[0]) * vx).astype(np.int32)
    iy = np.floor((y_valid - y_range[0]) / (y_range[1] - y_range[0]) * vy).astype(np.int32)
    iz = np.floor((z_valid - z_range[0]) / (z_range[1] - z_range[0]) * vz).astype(np.int32)

    ix = np.clip(ix, 0, vx - 1)
    iy = np.clip(iy, 0, vy - 1)
    iz = np.clip(iz, 0, vz - 1)

    occ[ix, iy, iz] = 1.0
    return occ


def occupancy_to_point_cloud(
    occupancy_grid: np.ndarray,
    threshold: float = 0.5,
    x_range: Tuple[float, float] = (VOXEL_X_MIN, VOXEL_X_MAX),
    y_range: Tuple[float, float] = (VOXEL_Y_MIN, VOXEL_Y_MAX),
    z_range: Tuple[float, float] = (VOXEL_Z_MIN, VOXEL_Z_MAX),
) -> np.ndarray:
    """Convert binary occupancy grid back into 3D voxel center coordinates.

    Args:
        occupancy_grid: Grid of shape (Vx, Vy, Vz).
        threshold: Binarization probability threshold.

    Returns:
        Array of shape [K, 3] containing active voxel center coordinates [x, y, z].
    """
    vx, vy, vz = occupancy_grid.shape
    active_indices = np.argwhere(occupancy_grid >= threshold)

    if len(active_indices) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Compute voxel center coordinates
    dx = (x_range[1] - x_range[0]) / vx
    dy = (y_range[1] - y_range[0]) / vy
    dz = (z_range[1] - z_range[0]) / vz

    xc = x_range[0] + (active_indices[:, 0] + 0.5) * dx
    yc = y_range[0] + (active_indices[:, 1] + 0.5) * dy
    zc = z_range[0] + (active_indices[:, 2] + 0.5) * dz

    return np.column_stack([xc, yc, zc]).astype(np.float32)
