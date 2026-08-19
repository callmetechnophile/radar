"""Deterministic Multi-Object Tracker and Evaluation Suite for VoD Radar Perception."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    """State of an active 3D object track."""

    def __init__(self, track_id: int, initial_box: np.ndarray, initial_class: int, initial_frame: int):
        self.track_id = track_id
        self.box = np.copy(initial_box)       # [x, y, z, l, w, h, yaw]
        self.cls = initial_class
        self.history = [initial_box]
        self.frames = [initial_frame]
        self.age = 1
        self.time_since_update = 0
        self.velocity = np.zeros(3, dtype=np.float32)

    def predict(self, dt: float = 0.07692):
        """Linear motion prediction."""
        self.box[:3] += self.velocity * dt
        self.age += 1
        self.time_since_update += 1

    def update(self, detected_box: np.ndarray, detected_class: int, frame_idx: int, dt: float = 0.07692):
        """Update track with new associated detection."""
        new_vel = (detected_box[:3] - self.box[:3]) / max(1e-4, dt)
        self.velocity = 0.6 * self.velocity + 0.4 * new_vel
        self.box = np.copy(detected_box)
        self.cls = detected_class
        self.history.append(self.box)
        self.frames.append(frame_idx)
        self.time_since_update = 0


class DeterministicVoDTracker:
    """Deterministic Multi-Object Tracker using 3D Distance Gating and Greedy Association."""

    def __init__(
        self,
        dist_threshold: float = 2.5,
        max_age: int = 3,
        dt: float = 0.07692,
    ) -> None:
        self.dist_threshold = float(dist_threshold)
        self.max_age = int(max_age)
        self.dt = float(dt)
        self.next_id = 1
        self.tracks: List[Track] = []

    def reset(self):
        self.next_id = 1
        self.tracks = []

    def step(self, detected_boxes: np.ndarray, detected_classes: np.ndarray, frame_idx: int) -> List[Tuple[int, np.ndarray, int]]:
        """Process detections for one frame and return active track assignments.

        Args:
            detected_boxes: [N_det, 7] array.
            detected_classes: [N_det] array.
            frame_idx: Frame index.

        Returns:
            List of (track_id, box, class).
        """
        # 1. Predict track locations
        for trk in self.tracks:
            trk.predict(self.dt)

        N_det = len(detected_boxes)
        N_trk = len(self.tracks)

        matched_tracks = set()
        matched_dets = set()

        if N_trk > 0 and N_det > 0:
            # Build cost matrix based on 3D center Euclidean distance
            cost_matrix = np.zeros((N_trk, N_det), dtype=np.float32)
            for i, trk in enumerate(self.tracks):
                for j in range(N_det):
                    dist = np.linalg.norm(trk.box[:3] - detected_boxes[j, :3])
                    # Penalize class mismatch
                    if trk.cls != detected_classes[j]:
                        dist += 5.0
                    cost_matrix[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.dist_threshold:
                    self.tracks[r].update(detected_boxes[c], int(detected_classes[c]), frame_idx, self.dt)
                    matched_tracks.add(r)
                    matched_dets.add(c)

        # 2. Initialize new tracks for unmatched detections
        for j in range(N_det):
            if j not in matched_dets:
                new_trk = Track(self.next_id, detected_boxes[j], int(detected_classes[j]), frame_idx)
                self.next_id += 1
                self.tracks.append(new_trk)

        # 3. Prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 4. Return active tracks updated in current frame
        active_outputs = []
        for t in self.tracks:
            if t.time_since_update == 0:
                active_outputs.append((t.track_id, t.box, t.cls))

        return active_outputs


def evaluate_tracking_hota_idf1_mota(
    pred_trajectories_by_frame: List[List[Tuple[int, np.ndarray, int]]],
    gt_trajectories_by_frame: List[List[Tuple[int, np.ndarray, int]]],
    dist_threshold: float = 2.0,
) -> Dict[str, float]:
    """Compute HOTA, IDF1, MOTA, ID Switches, Track Fragmentation, and Trajectory Errors.

    Args:
        pred_trajectories_by_frame: Per-frame list of (pred_id, box, class).
        gt_trajectories_by_frame: Per-frame list of (gt_id, box, class).
        dist_threshold: Maximum matching distance threshold in meters.

    Returns:
        Dictionary of tracking performance metrics.
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0
    fragmentations = 0

    id_mapping = {}  # gt_id -> last_pred_id
    pos_errors = []

    for f_idx in range(len(gt_trajectories_by_frame)):
        gts = gt_trajectories_by_frame[f_idx]
        preds = pred_trajectories_by_frame[f_idx]

        total_gt += len(gts)
        matched_gt = set()
        matched_pr = set()

        if len(gts) > 0 and len(preds) > 0:
            cost = np.zeros((len(gts), len(preds)), dtype=np.float32)
            for g_i, (g_id, g_box, g_cls) in enumerate(gts):
                for p_i, (p_id, p_box, p_cls) in enumerate(preds):
                    d = np.linalg.norm(g_box[:3] - p_box[:3])
                    cost[g_i, p_i] = d

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= dist_threshold:
                    g_id = gts[r][0]
                    p_id = preds[c][0]
                    matched_gt.add(r)
                    matched_pr.add(c)
                    total_tp += 1
                    pos_errors.append(cost[r, c])

                    # Check ID switch
                    if g_id in id_mapping:
                        if id_mapping[g_id] != p_id:
                            id_switches += 1
                            fragmentations += 1
                    id_mapping[g_id] = p_id

        total_fn += len(gts) - len(matched_gt)
        total_fp += len(preds) - len(matched_pr)

    # MOTA = 1 - (FN + FP + IDSW) / max(1, GT)
    mota = 1.0 - (total_fn + total_fp + id_switches) / max(1, total_gt)
    mota = max(0.0, mota)

    # IDF1 = 2 * TP / (2 * TP + FP + FN)
    idf1 = (2.0 * total_tp) / max(1, 2 * total_tp + total_fp + total_fn)

    # HOTA approximation = sqrt(DetA * AssA)
    det_a = total_tp / max(1, total_tp + total_fp + total_fn)
    ass_a = (total_tp - id_switches) / max(1, total_tp)
    ass_a = max(0.0, ass_a)
    hota = np.sqrt(det_a * ass_a)

    return {
        "HOTA": float(hota),
        "IDF1": float(idf1),
        "MOTA": float(mota),
        "id_switches": int(id_switches),
        "track_fragmentations": int(fragmentations),
        "mean_trajectory_error_m": float(np.mean(pos_errors)) if pos_errors else 0.0,
    }
