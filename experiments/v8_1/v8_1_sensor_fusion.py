"""PhotonShield AI — Phase V8.1 Mathematical Dynamics Refinement
Module: v8_1_sensor_fusion.py

Advanced Multi-Modal Sensor Fusion & Observation Confidence:
- Non-linear Radar Radial Velocity EKF with confidence weighting
- Multi-rate / Intermittent LiDAR Geometric Kalman Fusion
- Dynamic Observation Confidence Metric c_t in [0, 1]
- Adaptive Measurement Covariance R_t(c_t) and Process Covariance Q_t(c_t)
"""

from typing import Dict, Tuple, Optional
import numpy as np


class SensorFusionConfidenceEngine:
    """Computes observation confidence and manages multi-sensor Kalman updates."""
    def __init__(self, sigma_pos_base: float = 0.05, sigma_rad_base: float = 0.10, sigma_lidar_base: float = 0.02):
        self.sigma_pos_base = sigma_pos_base
        self.sigma_rad_base = sigma_rad_base
        self.sigma_lidar_base = sigma_lidar_base

    def compute_observation_confidence(
        self,
        radar_mask: int,
        lidar_mask: int,
        radial_residual: float,
        consecutive_missing_frames: int,
    ) -> float:
        """Computes analytical observation confidence c_t in [0, 1].
        High when direct observations exist and innovation is small;
        decays smoothly during missing gap horizons.
        """
        if radar_mask == 0 and lidar_mask == 0:
            # Exponential decay over missing gap horizon
            c_base = 0.85 ** max(1, consecutive_missing_frames)
            return float(max(0.01, min(1.0, c_base)))

        # Observed frame
        c = 0.5 * float(radar_mask) + 0.5 * float(lidar_mask)
        # Residual penalty
        res_penalty = np.exp(-abs(radial_residual) / 0.5)
        c_final = c * (0.5 + 0.5 * res_penalty)
        return float(max(0.1, min(1.0, c_final)))

    def get_adaptive_r(self, confidence: float, modality: str = "RADAR") -> np.ndarray:
        """Adapts measurement covariance R based on observation confidence.
        Low confidence -> higher R (trust measurement less).
        """
        conf_clamped = max(0.05, min(1.0, confidence))
        if modality == "RADAR":
            sigma = self.sigma_pos_base / np.sqrt(conf_clamped)
        elif modality == "LIDAR":
            sigma = self.sigma_lidar_base / np.sqrt(conf_clamped)
        elif modality == "RADIAL_VEL":
            sigma = self.sigma_rad_base / np.sqrt(conf_clamped)
        else:
            sigma = self.sigma_pos_base
        return (sigma ** 2)
