"""Timestamp utilities and temporal statistics for Oxford Radar RobotCar dataset."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def compute_timestamp_statistics(timestamps_us: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive temporal statistics from radar timestamps in microseconds.

    Args:
        timestamps_us: 1D array of timestamps in microseconds.

    Returns:
        Dictionary containing temporal distribution, jitter, and gap metrics.
    """
    if len(timestamps_us) < 2:
        return {
            "count": len(timestamps_us),
            "dt_min_s": 0.0,
            "dt_max_s": 0.0,
            "dt_mean_s": 0.0,
            "dt_median_s": 0.0,
            "dt_std_s": 0.0,
            "fps": 0.0,
            "jitter_s": 0.0,
            "largest_gap_s": 0.0,
            "total_duration_s": 0.0,
        }

    # Convert to seconds
    t_sec = timestamps_us.astype(np.float64) / 1e6
    dts = np.diff(t_sec)

    dt_min = float(np.min(dts))
    dt_max = float(np.max(dts))
    dt_mean = float(np.mean(dts))
    dt_median = float(np.median(dts))
    dt_std = float(np.std(dts))
    fps = 1.0 / dt_mean if dt_mean > 0 else 0.0
    jitter = dt_std
    largest_gap = dt_max
    total_duration = float(t_sec[-1] - t_sec[0])

    return {
        "count": len(timestamps_us),
        "dt_min_s": dt_min,
        "dt_max_s": dt_max,
        "dt_mean_s": dt_mean,
        "dt_median_s": dt_median,
        "dt_std_s": dt_std,
        "fps": fps,
        "jitter_s": jitter,
        "largest_gap_s": largest_gap,
        "total_duration_s": total_duration,
        "dts_s": dts.tolist(),
    }


def find_temporal_windows(
    timestamps_us: np.ndarray,
    window_length: int,
    max_allowed_gap_s: float = 0.50,
) -> Tuple[List[Tuple[int, int]], int]:
    """Construct valid sliding window sequence indices of length T with continuity checks.

    Args:
        timestamps_us: 1D array of timestamps in microseconds.
        window_length: Length T of sequences.
        max_allowed_gap_s: Maximum allowable gap between consecutive frames in seconds.

    Returns:
        Tuple of (list of valid (start_idx, end_idx) tuples, number of rejected windows).
    """
    valid_windows = []
    rejected_count = 0
    t_sec = timestamps_us.astype(np.float64) / 1e6

    n = len(timestamps_us)
    for i in range(n - window_length + 1):
        window_dts = np.diff(t_sec[i : i + window_length])
        if np.any(window_dts > max_allowed_gap_s) or np.any(window_dts <= 0):
            rejected_count += 1
        else:
            valid_windows.append((i, i + window_length))

    return valid_windows, rejected_count


def interpolate_odometry_pose(
    query_timestamp_us: int,
    odom_timestamps_us: np.ndarray,
    poses: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Interpolate 6-DoF pose [x, y, z, roll, pitch, yaw] at a specific query timestamp.

    Args:
        query_timestamp_us: Target radar timestamp in microseconds.
        odom_timestamps_us: Monotonically increasing array of odometry timestamps.
        poses: Array of shape [N, 6] representing [x, y, z, roll, pitch, yaw].

    Returns:
        Tuple of (interpolated pose [6], interpolation time error in seconds).
    """
    idx = np.searchsorted(odom_timestamps_us, query_timestamp_us)
    if idx == 0:
        err = abs(query_timestamp_us - odom_timestamps_us[0]) / 1e6
        return poses[0].copy(), err
    if idx >= len(odom_timestamps_us):
        err = abs(query_timestamp_us - odom_timestamps_us[-1]) / 1e6
        return poses[-1].copy(), err

    t0 = odom_timestamps_us[idx - 1]
    t1 = odom_timestamps_us[idx]
    alpha = (query_timestamp_us - t0) / float(t1 - t0) if t1 > t0 else 0.0

    p0 = poses[idx - 1]
    p1 = poses[idx]

    interp_pose = (1.0 - alpha) * p0 + alpha * p1
    interp_err = min(abs(query_timestamp_us - t0), abs(query_timestamp_us - t1)) / 1e6

    return interp_pose, interp_err
