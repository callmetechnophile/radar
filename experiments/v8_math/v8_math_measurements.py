"""PhotonShield AI — V8-MATH Mathematical State Estimation Foundation
Module: v8_math_measurements.py

Radar measurement models, radial velocity physics, and observation mask handling.
Explicitly separates global and root-relative coordinate representations.
"""

from typing import Dict, Tuple, Optional
import numpy as np


class RadarMeasurementModel:
    """Rigorous mathematical radar measurement model.
    
    State: s_t = [p_t, v_t]^T or [p_t, v_t, a_t]^T
    Measurements:
        1. Geometric centroid/joint positions: y_pos = p_t + v_pos, v_pos ~ N(0, R_pos)
        2. Radial velocity: v_r = r_hat^T v = (p^T v) / ||p|| + v_rad, v_rad ~ N(0, R_rad)
        3. Observation mask: m_t in {0, 1}
    """
    def __init__(self, sigma_pos: float = 0.05, sigma_rad: float = 0.10):
        self.sigma_pos = float(sigma_pos)
        self.sigma_rad = float(sigma_rad)
        self.R_pos = (sigma_pos ** 2) * np.eye(3, dtype=np.float64)
        self.R_rad = np.array([[sigma_rad ** 2]], dtype=np.float64)

    @staticmethod
    def compute_radial_velocity(p: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> float:
        """Computes true radial velocity: v_r = (p^T v) / ||p||.
        Positive = receding (increasing range), Negative = approaching.
        """
        r = np.linalg.norm(p)
        if r < eps:
            return 0.0
        r_hat = p / r
        return float(np.dot(r_hat, v))

    @staticmethod
    def compute_radial_velocity_jacobian(p: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Computes analytic Jacobians:
        d(v_r)/dp = v / ||p|| - (p^T v) * p / ||p||^3
        d(v_r)/dv = p^T / ||p|| = r_hat^T
        Returns:
            H_p: shape (1, 3)
            H_v: shape (1, 3)
        """
        r = np.linalg.norm(p)
        if r < eps:
            return np.zeros((1, 3), dtype=np.float64), np.zeros((1, 3), dtype=np.float64)
        r_hat = p / r
        dot_pv = float(np.dot(p, v))
        H_p = (v / r - (dot_pv / (r ** 3)) * p).reshape(1, 3)
        H_v = r_hat.reshape(1, 3)
        return H_p, H_v

    def compute_radial_residual(self, v_r_meas: float, p_pred: np.ndarray, v_pred: np.ndarray) -> float:
        """Computes radial velocity innovation/residual: e_r = v_r,meas - r_hat^T v_pred."""
        v_r_pred = self.compute_radial_velocity(p_pred, v_pred)
        return float(v_r_meas - v_r_pred)

    @staticmethod
    def split_global_root_relative(joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separates global coordinates into root position and root-relative joint offsets.
        Input: joints [..., 22, 3]
        Returns:
            root: [..., 3]
            rel_joints: [..., 22, 3] (with root at index 0 equal to 0)
        """
        root = joints[..., 0:1, :].copy() # [..., 1, 3]
        rel_joints = joints - root
        return np.squeeze(root, axis=-2), rel_joints

    @staticmethod
    def compose_global_coordinates(root: np.ndarray, rel_joints: np.ndarray) -> np.ndarray:
        """Composes global joint coordinates from root position and root-relative offsets."""
        return rel_joints + root[..., np.newaxis, :]
