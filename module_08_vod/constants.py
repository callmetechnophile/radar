"""Constants, physical parameters, and voxel grid definitions for VoD Phase V6."""

from pathlib import Path

# Paths
VOD_DATASET_ROOT = Path(r"C:\Users\worka\research\photonpinn\vod\view_of_delft_PUBLIC")
RADAR_TRAIN_DIR = VOD_DATASET_ROOT / "radar" / "training" / "velodyne"
LIDAR_TRAIN_DIR = VOD_DATASET_ROOT / "lidar" / "training" / "velodyne"
CALIB_RADAR_DIR = VOD_DATASET_ROOT / "radar" / "training" / "calib"
CALIB_LIDAR_DIR = VOD_DATASET_ROOT / "lidar" / "training" / "calib"
LABEL_TRAIN_DIR = VOD_DATASET_ROOT / "lidar" / "training" / "label_2"
POSE_TRAIN_DIR = VOD_DATASET_ROOT / "lidar" / "training" / "pose"
IMAGESETS_DIR = VOD_DATASET_ROOT / "lidar" / "ImageSets"

# Native Radar Specifications
RADAR_POINT_CHANNELS = 7
RADAR_FIELD_NAMES = [
    "x",
    "y",
    "z",
    "rcs",
    "v_r",
    "v_r_compensated",
    "time_id",
]

# Temporal Sampling Rate
NOMINAL_FPS = 13.0
DT_NOMINAL = 1.0 / NOMINAL_FPS  # ~0.07692 s (76.92 ms)

# 3D Occupancy Representation Grid Parameters (Radar Coordinate Frame)
# Front-facing field of view: X in [0, 32] m, Y in [-16, 16] m, Z in [-2.5, 2.5] m
VOXEL_X_MIN = 0.0
VOXEL_X_MAX = 32.0
VOXEL_Y_MIN = -16.0
VOXEL_Y_MAX = 16.0
VOXEL_Z_MIN = -2.5
VOXEL_Z_MAX = 2.5

VOXEL_DIM_X = 32  # 1.0 m per voxel along X
VOXEL_DIM_Y = 32  # 1.0 m per voxel along Y
VOXEL_DIM_Z = 8   # 0.625 m per voxel along Z
TOTAL_VOXELS = VOXEL_DIM_X * VOXEL_DIM_Y * VOXEL_DIM_Z  # 8,192 binary cells

# Feature Dimensions
POINT_EMBED_DIM = 64
MAMBA_HIDDEN_DIM = 64
SEQUENCE_LENGTH_DEFAULT = 8
