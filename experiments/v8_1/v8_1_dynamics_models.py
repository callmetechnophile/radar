"""PhotonShield AI — Phase V8.1 Mathematical Dynamics Refinement
Module: v8_1_dynamics_models.py

Dynamical Motion Models:
- Model A: Constant Velocity (CV)
- Model B: Constant Acceleration (CA)
- Model C: Jerk-Limited / Damped Velocity Motion
- Model D: Nearly-Constant Jerk (NCJ)
- Model E: Adaptive-Q Kalman Model (Q_t dynamically conditioned on motion & gap)
"""

import math
from typing import Dict, Tuple, Optional, List
import numpy as np


class DynamicalMotionModels:
    """Provides discrete-time state-space matrices F and process noise Q for multiple motion models."""
    def __init__(self, dt: float = 0.033333, num_joints: int = 22):
        self.dt = float(dt)
        self.num_joints = num_joints

    def get_cv_model(self, sigma_a: float = 2.0) -> Tuple[np.ndarray, np.ndarray, int]:
        """Model A: Constant Velocity (state dim = 6 per joint)."""
        dt = self.dt
        F1 = np.eye(6, dtype=np.float64)
        F1[0:3, 3:6] = dt * np.eye(3)

        q_block = np.zeros((6, 6), dtype=np.float64)
        q_block[0:3, 0:3] = (dt ** 3 / 3.0) * np.eye(3)
        q_block[0:3, 3:6] = (dt ** 2 / 2.0) * np.eye(3)
        q_block[3:6, 0:3] = (dt ** 2 / 2.0) * np.eye(3)
        q_block[3:6, 3:6] = dt * np.eye(3)
        Q1 = (sigma_a ** 2) * q_block

        F = np.kron(np.eye(self.num_joints, dtype=np.float64), F1)
        Q = np.kron(np.eye(self.num_joints, dtype=np.float64), Q1)
        return F, Q, 6

    def get_ca_model(self, sigma_j: float = 5.0) -> Tuple[np.ndarray, np.ndarray, int]:
        """Model B: Constant Acceleration (state dim = 9 per joint)."""
        dt = self.dt
        F1 = np.eye(9, dtype=np.float64)
        F1[0:3, 3:6] = dt * np.eye(3)
        F1[0:3, 6:9] = 0.5 * (dt ** 2) * np.eye(3)
        F1[3:6, 6:9] = dt * np.eye(3)

        q_block = np.zeros((9, 9), dtype=np.float64)
        q_block[0:3, 0:3] = (dt ** 5 / 20.0) * np.eye(3)
        q_block[0:3, 3:6] = (dt ** 4 / 8.0) * np.eye(3)
        q_block[0:3, 6:9] = (dt ** 3 / 6.0) * np.eye(3)
        q_block[3:6, 0:3] = (dt ** 4 / 8.0) * np.eye(3)
        q_block[3:6, 3:6] = (dt ** 3 / 3.0) * np.eye(3)
        q_block[3:6, 6:9] = (dt ** 2 / 2.0) * np.eye(3)
        q_block[6:9, 0:3] = (dt ** 3 / 6.0) * np.eye(3)
        q_block[6:9, 3:6] = (dt ** 2 / 2.0) * np.eye(3)
        q_block[6:9, 6:9] = dt * np.eye(3)
        Q1 = (sigma_j ** 2) * q_block

        F = np.kron(np.eye(self.num_joints, dtype=np.float64), F1)
        Q = np.kron(np.eye(self.num_joints, dtype=np.float64), Q1)
        return F, Q, 9

    def get_jerk_limited_model(self, damping: float = 0.98, sigma_a: float = 2.0) -> Tuple[np.ndarray, np.ndarray, int]:
        """Model C: Jerk-Limited / Velocity-Damped Kinematic Model.
        Decays velocity slightly during missing observation horizons to prevent divergence.
        """
        dt = self.dt
        F1 = np.eye(6, dtype=np.float64)
        F1[0:3, 3:6] = dt * np.eye(3)
        F1[3:6, 3:6] = damping * np.eye(3) # velocity damping factor

        q_block = np.zeros((6, 6), dtype=np.float64)
        q_block[0:3, 0:3] = (dt ** 3 / 3.0) * np.eye(3)
        q_block[0:3, 3:6] = (dt ** 2 / 2.0) * np.eye(3)
        q_block[3:6, 0:3] = (dt ** 2 / 2.0) * np.eye(3)
        q_block[3:6, 3:6] = dt * np.eye(3)
        Q1 = (sigma_a ** 2) * q_block

        F = np.kron(np.eye(self.num_joints, dtype=np.float64), F1)
        Q = np.kron(np.eye(self.num_joints, dtype=np.float64), Q1)
        return F, Q, 6

    def get_ncj_model(self, sigma_snap: float = 10.0) -> Tuple[np.ndarray, np.ndarray, int]:
        """Model D: Nearly-Constant Jerk (state dim = 12 per joint: p, v, a, j)."""
        dt = self.dt
        F1 = np.eye(12, dtype=np.float64)
        F1[0:3, 3:6] = dt * np.eye(3)
        F1[0:3, 6:9] = 0.5 * (dt ** 2) * np.eye(3)
        F1[0:3, 9:12] = (dt ** 3 / 6.0) * np.eye(3)
        F1[3:6, 6:9] = dt * np.eye(3)
        F1[3:6, 9:12] = 0.5 * (dt ** 2) * np.eye(3)
        F1[6:9, 9:12] = dt * np.eye(3)

        q_block = np.zeros((12, 12), dtype=np.float64)
        q_block[0:3, 0:3] = (dt ** 7 / 252.0) * np.eye(3)
        q_block[3:6, 3:6] = (dt ** 5 / 20.0) * np.eye(3)
        q_block[6:9, 6:9] = (dt ** 3 / 3.0) * np.eye(3)
        q_block[9:12, 9:12] = dt * np.eye(3)
        Q1 = (sigma_snap ** 2) * q_block

        F = np.kron(np.eye(self.num_joints, dtype=np.float64), F1)
        Q = np.kron(np.eye(self.num_joints, dtype=np.float64), Q1)
        return F, Q, 12

    def compute_adaptive_q(
        self,
        base_Q: np.ndarray,
        vel_mag: float,
        innovation_norm: float,
        gap_horizon: int,
        confidence: float,
    ) -> np.ndarray:
        """Model E: Adaptive process noise covariance Q_t.
        Q_t scales up during agile maneuvers, high innovation, and longer gap horizons,
        and scales down during confident, steady tracking.
        """
        # Motion intensity multiplier
        alpha_vel = 1.0 + 0.5 * min(5.0, vel_mag)
        # Innovation multiplier
        alpha_innov = 1.0 + 0.3 * min(5.0, innovation_norm)
        # Gap expansion factor (grows uncertainty during missing horizon)
        alpha_gap = 1.0 + 0.1 * min(32, gap_horizon)
        # Confidence discount (higher confidence -> lower process drift)
        alpha_conf = 1.0 / (0.2 + 0.8 * max(0.0, min(1.0, confidence)))

        scale = alpha_vel * alpha_innov * alpha_gap * alpha_conf
        return base_Q * scale
