"""PhotonShield AI Module 08: View-of-Delft (VoD) 3D Radar Perception Foundation."""

from module_08_vod.constants import (
    RADAR_POINT_CHANNELS,
    RADAR_FIELD_NAMES,
    NOMINAL_FPS,
    DT_NOMINAL,
    VOXEL_DIM_X,
    VOXEL_DIM_Y,
    VOXEL_DIM_Z,
    TOTAL_VOXELS,
    POINT_EMBED_DIM,
    MAMBA_HIDDEN_DIM,
    SEQUENCE_LENGTH_DEFAULT,
)
from module_08_vod.radar_loader import (
    load_radar_point_cloud,
    load_lidar_point_cloud,
    load_calibration_txt,
    transform_lidar_to_radar,
    point_cloud_to_occupancy,
    occupancy_to_point_cloud,
)
from module_08_vod.radar_point_encoder import RadarPointEncoder
from module_08_vod.reconstruction_head import OccupancyReconstructionHead
from module_08_vod.temporal_model import VoDFramewiseBaseline, VoDMambaTemporalModel
from module_08_vod.sequence_builder import (
    extract_continuous_snippets,
    build_100_sequence_split,
    compute_training_normalization,
    VoDSequenceDataset,
)
from module_08_vod.losses import OccupancyReconstructionLoss
from module_08_vod.metrics import (
    compute_occupancy_iou_precision_recall,
    compute_chamfer_distance,
    compute_reconstruction_mse,
    compute_temporal_consistency,
    evaluate_batch_metrics,
)
from module_08_vod.visualization import plot_3d_and_bev_comparison
from module_08_vod.diagnostics import check_physical_plausibility, audit_model_edge_footprint

__all__ = [
    "RADAR_POINT_CHANNELS",
    "RADAR_FIELD_NAMES",
    "NOMINAL_FPS",
    "DT_NOMINAL",
    "VOXEL_DIM_X",
    "VOXEL_DIM_Y",
    "VOXEL_DIM_Z",
    "TOTAL_VOXELS",
    "POINT_EMBED_DIM",
    "MAMBA_HIDDEN_DIM",
    "SEQUENCE_LENGTH_DEFAULT",
    "load_radar_point_cloud",
    "load_lidar_point_cloud",
    "load_calibration_txt",
    "transform_lidar_to_radar",
    "point_cloud_to_occupancy",
    "occupancy_to_point_cloud",
    "RadarPointEncoder",
    "OccupancyReconstructionHead",
    "VoDFramewiseBaseline",
    "VoDMambaTemporalModel",
    "extract_continuous_snippets",
    "build_100_sequence_split",
    "compute_training_normalization",
    "VoDSequenceDataset",
    "OccupancyReconstructionLoss",
    "compute_occupancy_iou_precision_recall",
    "compute_chamfer_distance",
    "compute_reconstruction_mse",
    "compute_temporal_consistency",
    "evaluate_batch_metrics",
    "plot_3d_and_bev_comparison",
    "check_physical_plausibility",
    "audit_model_edge_footprint",
]
