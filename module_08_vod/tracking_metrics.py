"""Tracking and 3D Bounding-Box Evaluation Metrics for Phase V6.2."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.metrics import f1_score


def compute_3d_box_center_and_dim_mae(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute Center MAE (m), Dimension MAE (m), and Yaw MAE (rad).

    Box Format: [x, y, z, l, w, h, yaw]
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0

    # Center MAE: ||[x,y,z]_pred - [x,y,z]_gt||
    center_errors = np.linalg.norm(pred_boxes[:, :3] - gt_boxes[:, :3], axis=1)
    center_mae = float(np.mean(center_errors))

    # Dimension MAE: |l-l| + |w-w| + |h-h|
    dim_errors = np.mean(np.abs(pred_boxes[:, 3:6] - gt_boxes[:, 3:6]), axis=1)
    dim_mae = float(np.mean(dim_errors))

    # Yaw MAE (wrapped to [-pi, pi])
    yaw_diff = np.abs(pred_boxes[:, 6] - gt_boxes[:, 6])
    yaw_diff = np.minimum(yaw_diff, 2 * np.pi - yaw_diff)
    yaw_mae = float(np.mean(yaw_diff))

    return center_mae, dim_mae, yaw_mae


def compute_bev_and_3d_iou(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
) -> Tuple[float, float]:
    """Compute axis-aligned BEV IoU and 3D bounding box IoU approximation."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0

    ious_bev = []
    ious_3d = []

    for i in range(len(pred_boxes)):
        pb = pred_boxes[i]
        gb = gt_boxes[i]

        # 2D BEV Box bounds [x_min, x_max, y_min, y_max]
        px_min, px_max = pb[0] - pb[3] / 2.0, pb[0] + pb[3] / 2.0
        py_min, py_max = pb[1] - pb[4] / 2.0, pb[1] + pb[4] / 2.0
        gx_min, gx_max = gb[0] - gb[3] / 2.0, gb[0] + gb[3] / 2.0
        gy_min, gy_max = gb[1] - gb[4] / 2.0, gb[1] + gb[4] / 2.0

        ix_min = max(px_min, gx_min)
        ix_max = min(px_max, gx_max)
        iy_min = max(py_min, gy_min)
        iy_max = min(py_max, gy_max)

        inter_bev = max(0.0, ix_max - ix_min) * max(0.0, iy_max - iy_min)
        union_bev = (pb[3] * pb[4]) + (gb[3] * gb[4]) - inter_bev
        iou_bev = inter_bev / max(1e-6, union_bev)
        ious_bev.append(iou_bev)

        # 3D Box height overlap
        pz_min, pz_max = pb[2] - pb[5] / 2.0, pb[2] + pb[5] / 2.0
        gz_min, gz_max = gb[2] - gb[5] / 2.0, gb[2] + gb[5] / 2.0
        iz_min = max(pz_min, gz_min)
        iz_max = min(pz_max, gz_max)
        inter_z = max(0.0, iz_max - iz_min)

        vol_p = pb[3] * pb[4] * pb[5]
        vol_g = gb[3] * gb[4] * gb[5]
        inter_3d = inter_bev * inter_z
        union_3d = vol_p + vol_g - inter_3d
        iou_3d = inter_3d / max(1e-6, union_3d)
        ious_3d.append(iou_3d)

    return float(np.mean(ious_bev)), float(np.mean(ious_3d))


def compute_track_consistency_error(
    pred_trajectories: List[np.ndarray],
) -> float:
    """Compute temporal velocity smoothness error across sequential track predictions."""
    if len(pred_trajectories) == 0:
        return 0.0

    smoothness_errors = []
    for traj in pred_trajectories:
        if len(traj) >= 3:
            vel = traj[1:] - traj[:-1]
            acc = vel[1:] - vel[:-1]
            smoothness_errors.append(float(np.mean(np.linalg.norm(acc, axis=1))))

    return float(np.mean(smoothness_errors)) if smoothness_errors else 0.0
