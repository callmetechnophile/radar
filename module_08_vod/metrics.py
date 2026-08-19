"""Comprehensive 3D Spatial and Temporal Perception Metrics for VoD."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch

from module_08_vod.radar_loader import occupancy_to_point_cloud


def compute_occupancy_iou_precision_recall(
    pred_probs: np.ndarray,
    gt_occupancy: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute Intersection-over-Union (IoU), Precision, and Recall for binary 3D occupancy.

    Args:
        pred_probs: Predicted occupancy probabilities in [0, 1].
        gt_occupancy: Ground truth binary occupancy in {0, 1}.
        threshold: Classification binarization threshold.

    Returns:
        (iou, precision, recall) as floats in [0, 1].
    """
    pred_bin = (pred_probs >= threshold).astype(bool)
    gt_bin = (gt_occupancy >= 0.5).astype(bool)

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
    fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()

    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return iou, precision, recall


def compute_chamfer_distance(
    pts_pred: np.ndarray,
    pts_gt: np.ndarray,
) -> Tuple[float, float]:
    """Compute bidirectional Chamfer Distance (m) and Point-to-Surface error (m).

    Args:
        pts_pred: Array of shape [N, 3] of predicted 3D coordinates.
        pts_gt: Array of shape [M, 3] of ground-truth 3D coordinates.

    Returns:
        (chamfer_distance, point_to_surface_error) in meters.
    """
    if len(pts_pred) == 0 and len(pts_gt) == 0:
        return 0.0, 0.0
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return 32.0, 32.0  # Max bounding box penalty

    # Efficient pairwise squared Euclidean distances
    # ||A - B||^2 = ||A||^2 + ||B||^2 - 2 A B^T
    dists = np.sum(pts_pred**2, axis=1, keepdims=True) + np.sum(pts_gt**2, axis=1, keepdims=True).T - 2 * np.dot(pts_pred, pts_gt.T)
    dists = np.maximum(dists, 0.0)

    min_dist_pred_to_gt = np.sqrt(np.min(dists, axis=1))  # [N]
    min_dist_gt_to_pred = np.sqrt(np.min(dists, axis=0))  # [M]

    p2s_error = float(np.mean(min_dist_pred_to_gt))
    chamfer_dist = float(np.mean(min_dist_pred_to_gt) + np.mean(min_dist_gt_to_pred)) / 2.0

    return chamfer_dist, p2s_error


def compute_reconstruction_mse(
    pred_probs: np.ndarray,
    gt_occupancy: np.ndarray,
) -> float:
    """Compute voxel-wise Mean Squared Error."""
    return float(np.mean((pred_probs - gt_occupancy) ** 2))


def compute_temporal_consistency(
    pred_probs_seq: np.ndarray,
) -> float:
    """Compute frame-to-frame temporal change smoothness in predicted occupancy.

    Args:
        pred_probs_seq: Array of shape [T, Vx, Vy, Vz].

    Returns:
        Mean absolute frame-to-frame difference (lower indicates smoother continuity).
    """
    if len(pred_probs_seq) <= 1:
        return 0.0
    diffs = np.abs(pred_probs_seq[1:] - pred_probs_seq[:-1])
    return float(np.mean(diffs))


def evaluate_batch_metrics(
    pred_logits: torch.Tensor,
    gt_occupancy: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all evaluation metrics across a batch of predicted and ground truth occupancy grids."""
    probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
    gt = gt_occupancy.detach().cpu().numpy()

    ious = []
    precisions = []
    recalls = []
    mses = []
    chamfers = []
    p2s_errors = []
    temp_consistencies = []

    B = probs.shape[0]
    T = probs.shape[1] if probs.ndim == 5 else 1

    for b in range(B):
        probs_b = probs[b]  # [T, Vx, Vy, Vz] or [Vx, Vy, Vz]
        gt_b = gt[b]

        if probs.ndim == 5:
            temp_consistencies.append(compute_temporal_consistency(probs_b))
            for t in range(T):
                iou, prec, rec = compute_occupancy_iou_precision_recall(probs_b[t], gt_b[t], threshold)
                ious.append(iou)
                precisions.append(prec)
                recalls.append(rec)
                mses.append(compute_reconstruction_mse(probs_b[t], gt_b[t]))

                pts_p = occupancy_to_point_cloud(probs_b[t], threshold)
                pts_g = occupancy_to_point_cloud(gt_b[t], 0.5)
                cd, p2s = compute_chamfer_distance(pts_p, pts_g)
                chamfers.append(cd)
                p2s_errors.append(p2s)
        else:
            iou, prec, rec = compute_occupancy_iou_precision_recall(probs_b, gt_b, threshold)
            ious.append(iou)
            precisions.append(prec)
            recalls.append(rec)
            mses.append(compute_reconstruction_mse(probs_b, gt_b))

            pts_p = occupancy_to_point_cloud(probs_b, threshold)
            pts_g = occupancy_to_point_cloud(gt_b, 0.5)
            cd, p2s = compute_chamfer_distance(pts_p, pts_g)
            chamfers.append(cd)
            p2s_errors.append(p2s)

    return {
        "occupancy_iou": float(np.mean(ious)),
        "occupancy_precision": float(np.mean(precisions)),
        "occupancy_recall": float(np.mean(recalls)),
        "reconstruction_mse": float(np.mean(mses)),
        "chamfer_distance_m": float(np.mean(chamfers)),
        "point_to_surface_m": float(np.mean(p2s_errors)),
        "temporal_consistency": float(np.mean(temp_consistencies)) if temp_consistencies else 0.0,
    }
