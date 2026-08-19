"""Unit tests for DeterministicVoDTracker and HOTA/IDF1 evaluation."""

import numpy as np
import pytest
from module_08_vod.tracker import DeterministicVoDTracker, evaluate_tracking_hota_idf1_mota


def test_deterministic_tracker_step():
    """Verify track state update and persistence."""
    tracker = DeterministicVoDTracker(dist_threshold=2.5, max_age=3)

    # Frame 0: 2 detections
    det_boxes_f0 = np.array([
        [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        [15.0, 3.0, 0.0, 1.0, 0.5, 1.7, 0.0],
    ], dtype=np.float32)
    det_cls_f0 = np.array([0, 1])

    tracks_f0 = tracker.step(det_boxes_f0, det_cls_f0, frame_idx=0)
    assert len(tracks_f0) == 2
    assert tracks_f0[0][0] == 1  # Track ID 1
    assert tracks_f0[1][0] == 2  # Track ID 2

    # Frame 1: Detections slightly displaced (moving forward)
    det_boxes_f1 = np.array([
        [10.8, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        [15.2, 3.0, 0.0, 1.0, 0.5, 1.7, 0.0],
    ], dtype=np.float32)
    det_cls_f1 = np.array([0, 1])

    tracks_f1 = tracker.step(det_boxes_f1, det_cls_f1, frame_idx=1)
    assert len(tracks_f1) == 2
    assert tracks_f1[0][0] == 1  # Preserved Track ID 1
    assert tracks_f1[1][0] == 2  # Preserved Track ID 2


def test_evaluate_tracking_metrics():
    """Verify HOTA, IDF1, and MOTA calculation."""
    gt_traj = [
        [(1, np.array([10.0, 0.0, 0.0]), 0)],
        [(1, np.array([11.0, 0.0, 0.0]), 0)],
        [(1, np.array([12.0, 0.0, 0.0]), 0)],
    ]
    pred_traj = [
        [(1, np.array([10.1, 0.0, 0.0]), 0)],
        [(1, np.array([11.1, 0.0, 0.0]), 0)],
        [(1, np.array([12.1, 0.0, 0.0]), 0)],
    ]

    res = evaluate_tracking_hota_idf1_mota(pred_traj, gt_traj, dist_threshold=2.0)
    assert res["HOTA"] > 0.8
    assert res["IDF1"] == 1.0
    assert res["MOTA"] == 1.0
    assert res["id_switches"] == 0
