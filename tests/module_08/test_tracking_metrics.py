"""Unit tests for tracking and 3D bounding box metrics."""

import numpy as np
import pytest
from module_08_vod.tracking_metrics import (
    compute_3d_box_center_and_dim_mae,
    compute_bev_and_3d_iou,
    compute_track_consistency_error,
)


def test_box_mae_and_iou():
    """Verify MAE and IoU computations on synthetic 3D bounding boxes."""
    # [x, y, z, l, w, h, yaw]
    gt_box = np.array([[10.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0]], dtype=np.float32)
    pred_box = np.array([[10.5, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32)

    c_mae, d_mae, y_mae = compute_3d_box_center_and_dim_mae(pred_box, gt_box)
    assert abs(c_mae - 0.5) < 1e-4
    assert d_mae == 0.0
    assert abs(y_mae - 0.1) < 1e-4

    iou_bev, iou_3d = compute_bev_and_3d_iou(pred_box, gt_box)
    assert 0.0 < iou_bev <= 1.0
    assert 0.0 < iou_3d <= 1.0


def test_track_consistency_error():
    """Verify trajectory smoothness computation."""
    # Constant velocity trajectory -> 0 acceleration error
    traj_smooth = [np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])]
    err_smooth = compute_track_consistency_error(traj_smooth)
    assert err_smooth == 0.0

    # Jittery trajectory -> positive error
    traj_jitter = [np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0], [3.0, 0.0]])]
    err_jitter = compute_track_consistency_error(traj_jitter)
    assert err_jitter > 0.0
