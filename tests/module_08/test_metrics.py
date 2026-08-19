"""Unit tests for 3D evaluation metrics."""

import numpy as np
import pytest
import torch
from module_08_vod.metrics import (
    compute_occupancy_iou_precision_recall,
    compute_chamfer_distance,
    compute_reconstruction_mse,
    compute_temporal_consistency,
    evaluate_batch_metrics,
)


def test_iou_precision_recall():
    """Verify IoU, precision, and recall math."""
    pred = np.array([
        [[1.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ])
    gt = np.array([
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ])

    iou, prec, rec = compute_occupancy_iou_precision_recall(pred, gt, threshold=0.5)
    # TP = 1, FP = 1, FN = 0 -> IoU = 1/2 = 0.5, Prec = 1/2 = 0.5, Rec = 1/1 = 1.0
    assert abs(iou - 0.5) < 1e-5
    assert abs(prec - 0.5) < 1e-5
    assert abs(rec - 1.0) < 1e-5


def test_chamfer_distance():
    """Verify Chamfer distance between identical and translated point sets."""
    pts1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    pts2 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    cd, p2s = compute_chamfer_distance(pts1, pts2)
    assert cd == 0.0
    assert p2s == 0.0

    pts_shift = pts1 + 2.0
    cd_shift, p2s_shift = compute_chamfer_distance(pts1, pts_shift)
    assert cd_shift > 0.0


def test_evaluate_batch_metrics():
    """Verify batch evaluation wrapper."""
    logits = torch.randn(2, 4, 16, 16, 8)
    gt = (torch.rand(2, 4, 16, 16, 8) > 0.8).float()

    m = evaluate_batch_metrics(logits, gt, threshold=0.5)
    assert "occupancy_iou" in m
    assert "reconstruction_mse" in m
    assert "chamfer_distance_m" in m
    assert "point_to_surface_m" in m
    assert "temporal_consistency" in m
