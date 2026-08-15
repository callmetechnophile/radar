"""Data structures for native Oxford radar frames and temporal sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class RadarFrame:
    """Represents a single native Oxford Navtech radar scan.

    Attributes:
        radar: 2D array of polar radar intensities of shape [num_azimuths, num_range_bins].
        timestamp_us: Timestamp of the radar scan in microseconds (int).
        azimuths: 1D array of azimuth angles in radians of length [num_azimuths].
        metadata: Dictionary containing range resolution, sensor origin, etc.
    """
    radar: np.ndarray
    timestamp_us: int
    azimuths: np.ndarray
    metadata: Dict[str, Any]

    @property
    def num_azimuths(self) -> int:
        return self.radar.shape[0]

    @property
    def num_range_bins(self) -> int:
        return self.radar.shape[1]

    @property
    def timestamp_s(self) -> float:
        return float(self.timestamp_us) / 1e6


@dataclass
class RadarSequence:
    """Represents a chronologically ordered sequence of radar frames.

    Attributes:
        frames: List of RadarFrame objects of length T.
        timestamps_us: 1D numpy array of timestamps in microseconds.
        dt: Average temporal interval between consecutive frames in seconds.
        odometry_poses: Optional array of synchronized 6-DoF poses [T, 6].
    """
    frames: List[RadarFrame]
    timestamps_us: np.ndarray
    dt: float
    odometry_poses: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def polar_tensor(self) -> np.ndarray:
        """Returns 3D array of stacked polar scans [T, num_azimuths, num_range_bins]."""
        return np.stack([f.radar for f in self.frames], axis=0)

    @property
    def timestamps_s(self) -> np.ndarray:
        return self.timestamps_us.astype(np.float64) / 1e6

    @property
    def total_duration_s(self) -> float:
        if len(self.timestamps_us) < 2:
            return 0.0
        return float((self.timestamps_us[-1] - self.timestamps_us[0]) / 1e6)
