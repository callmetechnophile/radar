"""PhotonShield AI — V8-MATH Mathematical State Estimation Foundation
Module: v8_math_lidar.py

Synchronized LiDAR measurement model and purely analytical sensor fusion.
No neural network, no learned weights — closed-form analytical Kalman fusion.
"""

from typing import Dict, Tuple, Optional
import numpy as np


class LidarMeasurementModel:
    """Analytical LiDAR geometric observation model.
    LiDAR provides direct 3D geometric surface/centroid observations:
        y_L = H_L s_t + v_L,  v_L ~ N(0, R_L)
    with substantially lower measurement noise covariance (e.g., sigma_L = 0.02m vs sigma_radar = 0.08m).
    """
    def __init__(self, sigma_lidar: float = 0.02, num_joints: int = 22, dim_per_joint: int = 6):
        self.sigma_lidar = float(sigma_lidar)
        self.num_joints = num_joints
        self.dpj = dim_per_joint
        self.state_dim = num_joints * dim_per_joint
        self.meas_dim = 3 * num_joints

        # Measurement matrix H_L extracting 3D positions
        self.H_L = np.zeros((self.meas_dim, self.state_dim), dtype=np.float64)
        for j in range(self.num_joints):
            self.H_L[j*3:(j+1)*3, j*self.dpj:j*self.dpj+3] = np.eye(3)

        self.R_L = (self.sigma_lidar ** 2) * np.eye(self.meas_dim, dtype=np.float64)

    def update_lidar(self, x_pred: np.ndarray, P_pred: np.ndarray, y_lidar: np.ndarray, mask: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
        """Performs analytical Kalman measurement update using LiDAR observations."""
        if mask == 0:
            return x_pred.copy(), P_pred.copy(), 0.0

        y_flat = y_lidar.reshape(-1)
        innov = y_flat - self.H_L @ x_pred
        S = self.H_L @ P_pred @ self.H_L.T + self.R_L
        K = P_pred @ self.H_L.T @ np.linalg.inv(S)

        x_up = x_pred + K @ innov
        I_KH = np.eye(self.state_dim, dtype=np.float64) - K @ self.H_L
        P_up = I_KH @ P_pred @ I_KH.T + K @ self.R_L @ K.T
        lidar_residual = float(np.mean(np.abs(innov)))
        return x_up, P_up, lidar_residual


class AnalyticalSensorFusion:
    """Analytical multi-modal fusion combining Radar and LiDAR observations."""
    def __init__(self, kf_radar, lidar_model: LidarMeasurementModel):
        self.kf_radar = kf_radar
        self.lidar = lidar_model

    def fuse_step(
        self, x_pred: np.ndarray, P_pred: np.ndarray,
        radar_meas: Optional[np.ndarray], lidar_meas: Optional[np.ndarray],
        radar_mask: int = 1, lidar_mask: int = 1,
        mode: str = "FUSED" # "RADAR_ONLY", "LIDAR_ONLY", "FUSED"
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Sequential optimal Kalman update for multi-rate / multi-modal sensor fusion."""
        x_curr = x_pred.copy()
        P_curr = P_pred.copy()
        residuals = {"radar_pos_residual": 0.0, "lidar_pos_residual": 0.0}

        # Radar update
        if mode in ("RADAR_ONLY", "FUSED") and radar_meas is not None and radar_mask == 1:
            x_curr, P_curr = self.kf_radar.update(x_curr, P_curr, radar_meas, mask=1)
            residuals["radar_pos_residual"] = float(np.mean(np.abs(radar_meas.reshape(-1) - self.kf_radar.H @ x_curr)))

        # LiDAR update (sequential linear Kalman update)
        if mode in ("LIDAR_ONLY", "FUSED") and lidar_meas is not None and lidar_mask == 1:
            x_curr, P_curr, l_res = self.lidar.update_lidar(x_curr, P_curr, lidar_meas, mask=1)
            residuals["lidar_pos_residual"] = l_res

        return x_curr, P_curr, residuals
