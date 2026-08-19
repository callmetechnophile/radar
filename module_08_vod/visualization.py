"""Visualization Utilities for Phase V6.1 VoD 3D Radar Perception."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from module_08_vod.radar_loader import occupancy_to_point_cloud


def plot_3d_and_bev_comparison(
    radar_pts: np.ndarray,
    gt_occ: np.ndarray,
    framewise_occ: np.ndarray,
    mamba_occ: np.ndarray,
    save_path: Path,
    frame_title: str = "Test Frame",
) -> None:
    """Generate 4-way comparative visualization (BEV top-down and front projection).

    Subplots:
    1. Input Native Radar Points
    2. Ground Truth LiDAR Occupancy
    3. Frame-Wise Baseline Reconstruction
    4. Mamba Temporal Model Reconstruction
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Native Radar Input
    ax = axes[0, 0]
    ax.scatter(radar_pts[:, 1], radar_pts[:, 0], c=radar_pts[:, 3], cmap="viridis", s=25, alpha=0.9)
    ax.set_title(f"1. Native Radar Input ({len(radar_pts)} pts, Colored by RCS)", fontweight="bold")
    ax.set_xlabel("Lateral Y (m) [Left +, Right -]")
    ax.set_ylabel("Longitudinal X (m) [Forward +]")
    ax.set_xlim(-16, 16)
    ax.set_ylim(0, 32)
    ax.grid(True, alpha=0.3)

    # 2. Ground Truth LiDAR Occupancy
    gt_pts = occupancy_to_point_cloud(gt_occ, threshold=0.5)
    ax = axes[0, 1]
    if len(gt_pts) > 0:
        ax.scatter(gt_pts[:, 1], gt_pts[:, 0], c=gt_pts[:, 2], cmap="plasma", s=15, alpha=0.85)
    ax.set_title(f"2. Ground Truth LiDAR Occupancy ({len(gt_pts)} active voxels)", fontweight="bold")
    ax.set_xlabel("Lateral Y (m)")
    ax.set_ylabel("Longitudinal X (m)")
    ax.set_xlim(-16, 16)
    ax.set_ylim(0, 32)
    ax.grid(True, alpha=0.3)

    # 3. Frame-Wise Baseline Reconstruction
    fw_pts = occupancy_to_point_cloud(framewise_occ, threshold=0.4)
    ax = axes[1, 0]
    if len(fw_pts) > 0:
        ax.scatter(fw_pts[:, 1], fw_pts[:, 0], c="#d62728", s=20, alpha=0.8)
    ax.set_title(f"3. Frame-Wise Baseline (No Temporal Prior, {len(fw_pts)} voxels)", fontweight="bold")
    ax.set_xlabel("Lateral Y (m)")
    ax.set_ylabel("Longitudinal X (m)")
    ax.set_xlim(-16, 16)
    ax.set_ylim(0, 32)
    ax.grid(True, alpha=0.3)

    # 4. Mamba Temporal Model Reconstruction
    mb_pts = occupancy_to_point_cloud(mamba_occ, threshold=0.4)
    ax = axes[1, 1]
    if len(mb_pts) > 0:
        ax.scatter(mb_pts[:, 1], mb_pts[:, 0], c="#2ca02c", s=20, alpha=0.8)
    ax.set_title(f"4. Temporal Mamba Model (Selective SSM, {len(mb_pts)} voxels)", fontweight="bold")
    ax.set_xlabel("Lateral Y (m)")
    ax.set_ylabel("Longitudinal X (m)")
    ax.set_xlim(-16, 16)
    ax.set_ylim(0, 32)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"PhotonShield V6.1: {frame_title} 3D Occupancy Perception", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
