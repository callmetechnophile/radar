"""Feature extraction and dimensionality reduction for Oxford Navtech radar scans."""

from __future__ import annotations

from typing import Dict, Any, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_07_temporal.temporal_sequence import RadarFrame


class OxfordRadarFeatureExtractor(nn.Module):
    """Extracts a calibrated, physically grounded 64-dimensional feature representation from a native Oxford polar radar scan (400 azimuths x 3768 range bins).

    Components of 64-D Feature Vector:
    - 32-D Range Profile: Multi-scale radial energy distribution across range bins (near, mid, far).
    - 16-D Azimuthal Distribution: Directional radar reflectivity power across 16 angular sectors (22.5 deg each).
    - 8-D Statistical Moments: Mean, variance, skewness, kurtosis, peak-to-average power ratio (PAPR), total energy, SNR, dynamic range.
    - 8-D High-Energy Centroids: Locations and intensities of top spatial reflectivity clusters.
    """

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def extract_from_numpy(self, polar_scan: np.ndarray) -> np.ndarray:
        """Extract 64-D feature vector from a numpy array [400, 3768]."""
        # 1. Range Profile (32 bins)
        # Average across all 400 azimuths, then pool 3768 range bins down to 32
        radial_profile = np.mean(polar_scan, axis=0)  # [3768]
        # Pool to 32 bins using adaptive average pooling
        chunk_size = len(radial_profile) // 32
        range_features = np.array([
            np.mean(radial_profile[i * chunk_size : (i + 1) * chunk_size]) for i in range(32)
        ], dtype=np.float32)

        # 2. Azimuthal Profile (16 sectors of 25 azimuths each = 22.5 deg)
        azimuth_profile = np.mean(polar_scan, axis=1)  # [400]
        az_chunk = len(azimuth_profile) // 16
        azimuth_features = np.array([
            np.mean(azimuth_profile[i * az_chunk : (i + 1) * az_chunk]) for i in range(16)
        ], dtype=np.float32)

        # 3. Statistical Moments (8 features)
        flat = polar_scan.flatten()
        mean_val = float(np.mean(flat))
        std_val = float(np.std(flat)) + 1e-6
        max_val = float(np.max(flat))
        min_val = float(np.min(flat))
        papr = (max_val**2) / (mean_val**2 + 1e-6)
        total_energy = float(np.sum(flat**2) / len(flat))
        snr_est = (max_val - mean_val) / std_val
        dyn_range = max_val - min_val

        stats_features = np.array([
            mean_val, std_val, max_val, min_val,
            papr * 0.01, total_energy, snr_est * 0.1, dyn_range
        ], dtype=np.float32)

        # 4. Spatial Peak Centroids (8 features: 4 radial peaks + 4 angular peaks)
        peak_range_indices = np.argsort(radial_profile)[-4:] / 3768.0
        peak_az_indices = np.argsort(azimuth_profile)[-4:] / 400.0
        centroid_features = np.concatenate([peak_range_indices, peak_az_indices]).astype(np.float32)

        # Concatenate 32 + 16 + 8 + 8 = 64
        feat_vec = np.concatenate([
            range_features,
            azimuth_features,
            stats_features,
            centroid_features,
        ]).astype(np.float32)

        # Normalize feature vector to zero mean, unit variance
        feat_vec = (feat_vec - np.mean(feat_vec)) / (np.std(feat_vec) + 1e-6)
        return feat_vec

    def extract_from_frame(self, frame: RadarFrame) -> np.ndarray:
        return self.extract_from_numpy(frame.radar)

    def extract_sequence_features(self, frames: list[RadarFrame]) -> np.ndarray:
        """Extract [T, 64] numpy array from a sequence of RadarFrames."""
        return np.stack([self.extract_from_frame(f) for f in frames], axis=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward pass if x is already a batch of features or polar scans."""
        return x
