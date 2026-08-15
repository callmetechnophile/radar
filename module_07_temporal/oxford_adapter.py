"""Official Oxford Radar RobotCar Dataset Adapter for PhotonShield Temporal Foundation."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from PIL import Image

from module_07_temporal.temporal_sequence import RadarFrame, RadarSequence
from module_07_temporal.timestamp_utils import compute_timestamp_statistics, find_temporal_windows, interpolate_odometry_pose


class OxfordRadarAdapter:
    """Validated Dataset Adapter for Oxford Radar RobotCar dataset.

    Preserves native polar radar representation (400 azimuths x 3768 range bins),
    synchronized odometry poses, and Cartesian 2D grid generation.

    Attributes:
        dataset_root: Path to the dataset root directory containing radar/ and timestamps.
        range_resolution: Navtech radar range resolution in meters per bin (default 0.0432 m).
        num_azimuths: Number of azimuth angle bins per complete 360-degree rotation (400).
        max_range_m: Maximum sensing range in meters (default ~162.78 m).
    """

    DEFAULT_PATHS = [
        Path("C:/Users/worka/research/photonpinn/data/oxford_radar_robotcar/small"),
        Path("C:/Users/worka/research/photonpinn/oxford_radar_robotcar_dataset_sample_small/2019-01-10-14-36-48-radar-oxford-10k-partial"),
        Path("data/oxford_radar_robotcar/small"),
    ]

    def __init__(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        range_resolution: float = 0.0432,
    ) -> None:
        if dataset_root is not None:
            self.dataset_root = Path(dataset_root)
        else:
            self.dataset_root = self._auto_locate_root()

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Oxford Radar dataset root not found: {self.dataset_root}")

        self.range_resolution = float(range_resolution)
        self.radar_dir = self.dataset_root / "radar"
        self.timestamps_file = self.dataset_root / "radar.timestamps"
        self.gt_dir = self.dataset_root / "gt"
        self.vo_dir = self.dataset_root / "vo"

        # Load timestamps
        self.timestamps_us = self._load_timestamps()
        self.num_scans = len(self.timestamps_us)

        # Cache odometry if available
        self.odometry_data = self._load_odometry()

        # Extract basic dimensions from first frame
        if self.num_scans > 0:
            sample_frame = self.load_frame(0)
            self.num_azimuths = sample_frame.num_azimuths
            self.num_range_bins = sample_frame.num_range_bins
            self.max_range_m = self.num_range_bins * self.range_resolution
        else:
            self.num_azimuths = 400
            self.num_range_bins = 3768
            self.max_range_m = 3768 * self.range_resolution

    def _auto_locate_root(self) -> Path:
        for p in self.DEFAULT_PATHS:
            if p.exists() and (p / "radar").exists():
                return p
        return self.DEFAULT_PATHS[0]

    def _load_timestamps(self) -> np.ndarray:
        """Load microsecond timestamps from radar.timestamps file."""
        if not self.timestamps_file.exists():
            # Fallback to sorted filenames
            if self.radar_dir.exists():
                files = sorted(f for f in os.listdir(self.radar_dir) if f.endswith(".png"))
                return np.array([int(f.replace(".png", "")) for f in files], dtype=np.int64)
            return np.array([], dtype=np.int64)

        ts_list = []
        with open(self.timestamps_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    ts_list.append(int(parts[0]))
        return np.array(ts_list, dtype=np.int64)

    def _load_odometry(self) -> Optional[Dict[str, np.ndarray]]:
        """Load ground truth or visual odometry if present."""
        odom_file = self.gt_dir / "radar_odometry.csv"
        if not odom_file.exists():
            odom_file = self.vo_dir / "vo.csv"
        if not odom_file.exists():
            return None

        ts_list = []
        poses = []
        with open(odom_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = int(row.get("source_radar_timestamp", row.get("source_timestamp", 0)))
                x = float(row.get("x", 0.0))
                y = float(row.get("y", 0.0))
                z = float(row.get("z", 0.0))
                roll = float(row.get("roll", 0.0))
                pitch = float(row.get("pitch", 0.0))
                yaw = float(row.get("yaw", 0.0))
                ts_list.append(ts)
                poses.append([x, y, z, roll, pitch, yaw])

        if not ts_list:
            return None

        return {
            "timestamps_us": np.array(ts_list, dtype=np.int64),
            "poses": np.array(poses, dtype=np.float64),
        }

    def get_timestamps(self) -> np.ndarray:
        """Returns 1D array of all radar timestamps in microseconds."""
        return self.timestamps_us.copy()

    def get_timestamp(self, index: int) -> int:
        """Returns timestamp in microseconds for a specific frame index."""
        if index < 0 or index >= self.num_scans:
            raise IndexError(f"Frame index {index} out of range [0, {self.num_scans})")
        return int(self.timestamps_us[index])

    def load_frame(self, index_or_timestamp: Union[int, np.integer]) -> RadarFrame:
        """Load and decode a single native Oxford Navtech radar PNG scan.

        Args:
            index_or_timestamp: Index (0..N-1) or timestamp in microseconds.

        Returns:
            RadarFrame object with decoded polar intensities, azimuths, and timestamp.
        """
        if index_or_timestamp in self.timestamps_us:
            ts = int(index_or_timestamp)
        else:
            idx = int(index_or_timestamp)
            if idx < 0 or idx >= self.num_scans:
                raise IndexError(f"Frame index {idx} out of range [0, {self.num_scans})")
            ts = int(self.timestamps_us[idx])

        file_path = self.radar_dir / f"{ts}.png"
        if not file_path.exists():
            raise FileNotFoundError(f"Radar scan file not found: {file_path}")

        raw_data = np.array(Image.open(file_path))

        # Navtech CTS350-X parsing (11-byte header per azimuth)
        # Bytes 0-7: uint64 ray timestamp
        # Bytes 8-9: uint16 azimuth encoder tick (0..5599)
        # Byte 10: uint8 valid flag
        # Bytes 11..end: range bin intensities (0..255)
        raw_azimuths = raw_data[:, 8:10].copy().view(np.uint16).squeeze()
        azimuths = (raw_azimuths.astype(np.float64) / 5600.0) * (2.0 * np.pi)
        intensities = raw_data[:, 11:].astype(np.float32) / 255.0

        metadata = {
            "timestamp_us": ts,
            "range_resolution_m": self.range_resolution,
            "max_range_m": intensities.shape[1] * self.range_resolution,
            "num_azimuths": intensities.shape[0],
            "num_range_bins": intensities.shape[1],
            "source_file": str(file_path),
        }

        return RadarFrame(
            radar=intensities,
            timestamp_us=ts,
            azimuths=azimuths,
            metadata=metadata,
        )

    def load_sequence(self, start_idx: int, sequence_length: int) -> RadarSequence:
        """Load a contiguous temporal sequence of T radar frames.

        Args:
            start_idx: Start frame index.
            sequence_length: Number of frames T.

        Returns:
            RadarSequence containing loaded RadarFrames and synchronized poses.
        """
        if start_idx < 0 or start_idx + sequence_length > self.num_scans:
            raise IndexError(
                f"Sequence range [{start_idx}, {start_idx+sequence_length}) out of range [0, {self.num_scans})"
            )

        frames = [self.load_frame(i) for i in range(start_idx, start_idx + sequence_length)]
        seq_ts = self.timestamps_us[start_idx : start_idx + sequence_length]
        dts = np.diff(seq_ts.astype(np.float64) / 1e6)
        mean_dt = float(np.mean(dts)) if len(dts) > 0 else 0.25

        # Synchronize odometry if available
        synced_poses = None
        if self.odometry_data is not None:
            poses_list = []
            for ts in seq_ts:
                p, _ = interpolate_odometry_pose(
                    ts, self.odometry_data["timestamps_us"], self.odometry_data["poses"]
                )
                poses_list.append(p)
            synced_poses = np.array(poses_list, dtype=np.float64)

        return RadarSequence(
            frames=frames,
            timestamps_us=seq_ts,
            dt=mean_dt,
            odometry_poses=synced_poses,
        )

    def get_cartesian_radar(
        self,
        polar_radar_or_frame: Union[RadarFrame, np.ndarray],
        resolution_m_per_pixel: float = 0.25,
        cart_size_pixels: int = 640,
        azimuths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert native polar radar scan (400 azimuths x 3768 range bins) to 2D Cartesian grid.

        Args:
            polar_radar_or_frame: RadarFrame or 2D array [num_azimuths, num_range_bins].
            resolution_m_per_pixel: Spatial resolution in meters per pixel (default 0.25 m).
            cart_size_pixels: Size of Cartesian image grid (width = height = cart_size_pixels).
            azimuths: Optional array of azimuth angles in radians.

        Returns:
            2D numpy array of shape [cart_size_pixels, cart_size_pixels] with float32 intensities.
        """
        if isinstance(polar_radar_or_frame, RadarFrame):
            polar = polar_radar_or_frame.radar
            az = polar_radar_or_frame.azimuths
        else:
            polar = polar_radar_or_frame
            az = azimuths if azimuths is not None else np.linspace(0, 2 * np.pi, polar.shape[0], endpoint=False)

        num_az, num_range = polar.shape
        half_size = cart_size_pixels / 2.0

        # Build Cartesian pixel coordinates centered at (0, 0)
        x_coords = (np.arange(cart_size_pixels) - half_size) * resolution_m_per_pixel
        y_coords = (half_size - np.arange(cart_size_pixels)) * resolution_m_per_pixel
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Convert to Polar (range in bins, azimuth in radians)
        ranges_m = np.sqrt(xx**2 + yy**2)
        range_bins = ranges_m / self.range_resolution
        thetas = np.arctan2(xx, yy) % (2.0 * np.pi)  # 0 at forward/up, clockwise

        # Map thetas to azimuth indices
        # Navtech azimuths are monotonically increasing
        az_indices = (thetas / (2.0 * np.pi) * num_az).astype(np.int32) % num_az
        r_indices = np.clip(range_bins.astype(np.int32), 0, num_range - 1)

        # Mask beyond maximum sensor range
        valid_mask = range_bins < num_range

        cartesian = np.zeros((cart_size_pixels, cart_size_pixels), dtype=np.float32)
        cartesian[valid_mask] = polar[az_indices[valid_mask], r_indices[valid_mask]]

        return cartesian

    def get_odometry(self, timestamp_us_or_index: Union[int, np.integer]) -> Optional[Dict[str, Any]]:
        """Retrieve 6-DoF odometry pose aligned with a radar scan.

        Args:
            timestamp_us_or_index: Timestamp in microseconds or frame index.

        Returns:
            Dictionary with pose [x, y, z, roll, pitch, yaw] and synchronization error.
        """
        if self.odometry_data is None:
            return None

        if timestamp_us_or_index in self.timestamps_us:
            ts = int(timestamp_us_or_index)
        else:
            idx = int(timestamp_us_or_index)
            ts = int(self.timestamps_us[idx])

        pose, err_s = interpolate_odometry_pose(
            ts, self.odometry_data["timestamps_us"], self.odometry_data["poses"]
        )

        return {
            "timestamp_us": ts,
            "pose": pose,
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
            "roll": float(pose[3]),
            "pitch": float(pose[4]),
            "yaw": float(pose[5]),
            "sync_error_s": err_s,
        }

    def get_sequence_statistics(self) -> Dict[str, Any]:
        """Compute complete temporal sequence and window statistics across the dataset."""
        temporal_stats = compute_timestamp_statistics(self.timestamps_us)

        window_stats = {}
        for T in [4, 8, 16]:
            windows, rejected = find_temporal_windows(self.timestamps_us, window_length=T)
            window_stats[f"T_{T}"] = {
                "window_length": T,
                "valid_sequences": len(windows),
                "rejected_sequences": rejected,
                "mean_duration_s": (T - 1) * temporal_stats["dt_mean_s"],
            }

        return {
            "temporal_statistics": temporal_stats,
            "window_statistics": window_stats,
            "total_radar_scans": self.num_scans,
            "range_resolution_m": self.range_resolution,
            "max_range_m": self.max_range_m,
            "num_azimuths": self.num_azimuths,
            "num_range_bins": self.num_range_bins,
            "has_odometry": self.odometry_data is not None,
        }
