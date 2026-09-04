"""PhotonShield AI — V8-MATH Mathematical State Estimation Foundation
Module: v8_math_kalman.py

Linear Kalman Filter (CV and CA models) and Extended Kalman Filter (with non-linear radial velocity).
Includes process noise Q estimation from training data and rigorous observation-mask gating.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
from experiments.v8_math.v8_math_measurements import RadarMeasurementModel


class KalmanMotionModel:
    """Mathematical linear state-space motion model."""
    def __init__(self, dt: float = 0.033333, model_type: str = "CV", num_joints: int = 22):
        self.dt = float(dt)
        self.model_type = model_type.upper()
        self.num_joints = num_joints

        if self.model_type == "CV":
            # State per joint: [x, y, z, vx, vy, vz]^T (dim=6 per joint -> 132 total)
            self.dim_per_joint = 6
            self.F_single = np.eye(6, dtype=np.float64)
            self.F_single[0:3, 3:6] = self.dt * np.eye(3, dtype=np.float64)
        elif self.model_type == "CA":
            # State per joint: [x, y, z, vx, vy, vz, ax, ay, az]^T (dim=9 per joint -> 198 total)
            self.dim_per_joint = 9
            self.F_single = np.eye(9, dtype=np.float64)
            self.F_single[0:3, 3:6] = self.dt * np.eye(3, dtype=np.float64)
            self.F_single[0:3, 6:9] = 0.5 * (self.dt ** 2) * np.eye(3, dtype=np.float64)
            self.F_single[3:6, 6:9] = self.dt * np.eye(3, dtype=np.float64)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.state_dim = self.dim_per_joint * self.num_joints
        # Block diagonal transition matrix across all joints
        self.F = np.kron(np.eye(self.num_joints, dtype=np.float64), self.F_single)

    def compute_continuous_white_noise_q(self, sigma_a: float) -> np.ndarray:
        """Computes discrete Wiener process acceleration process noise matrix Q."""
        dt = self.dt
        if self.model_type == "CV":
            q_block = np.zeros((6, 6), dtype=np.float64)
            q_block[0:3, 0:3] = (dt ** 3 / 3.0) * np.eye(3)
            q_block[0:3, 3:6] = (dt ** 2 / 2.0) * np.eye(3)
            q_block[3:6, 0:3] = (dt ** 2 / 2.0) * np.eye(3)
            q_block[3:6, 3:6] = dt * np.eye(3)
            q_single = (sigma_a ** 2) * q_block
        else: # CA
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
            q_single = (sigma_a ** 2) * q_block

        return np.kron(np.eye(self.num_joints, dtype=np.float64), q_single)


class LinearKalmanFilter:
    """Standard Linear Kalman Filter with observation-mask gating."""
    def __init__(self, motion_model: KalmanMotionModel, Q: np.ndarray, R_pos: float = 0.05):
        self.motion = motion_model
        self.Q = Q.copy()
        self.state_dim = motion_model.state_dim
        self.num_joints = motion_model.num_joints
        self.dpj = motion_model.dim_per_joint

        # Measurement matrix H for 3D joint positions (dim = 3*J)
        self.meas_dim = 3 * self.num_joints
        self.H = np.zeros((self.meas_dim, self.state_dim), dtype=np.float64)
        for j in range(self.num_joints):
            self.H[j*3:(j+1)*3, j*self.dpj:j*self.dpj+3] = np.eye(3)

        self.R = (R_pos ** 2) * np.eye(self.meas_dim, dtype=np.float64)

    def initialize(self, initial_joints: np.ndarray, initial_vels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Initializes state vector x0 and error covariance P0."""
        x0 = np.zeros(self.state_dim, dtype=np.float64)
        for j in range(self.num_joints):
            x0[j*self.dpj:j*self.dpj+3] = initial_joints[j]
            if initial_vels is not None:
                x0[j*self.dpj+3:j*self.dpj+6] = initial_vels[j]

        # Initial covariance: positions confident (0.01m), velocities uncertain (1.0 m/s)
        P0_single = np.eye(self.dpj, dtype=np.float64)
        P0_single[0:3, 0:3] *= 0.01 ** 2
        P0_single[3:6, 3:6] *= 1.0 ** 2
        if self.dpj == 9:
            P0_single[6:9, 6:9] *= 5.0 ** 2
        P0 = np.kron(np.eye(self.num_joints, dtype=np.float64), P0_single)
        return x0, P0

    def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Time update (prediction step): x_pred = F x, P_pred = F P F^T + Q."""
        x_pred = self.motion.F @ x
        P_pred = self.motion.F @ P @ self.motion.F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, y: np.ndarray, mask: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Measurement update with observation gating (m_t in {0, 1})."""
        if mask == 0:
            # GATED: Observation missing -> Prediction only! Exactly zero-shock elimination!
            return x_pred.copy(), P_pred.copy()

        # Innovation: v = y - H x_pred
        y_flat = y.reshape(-1)
        v = y_flat - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        x_up = x_pred + K @ v
        # Numerically stable Joseph form: P = (I - KH) P (I - KH)^T + K R K^T
        I_KH = np.eye(self.state_dim, dtype=np.float64) - K @ self.H
        P_up = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        return x_up, P_up

    def extract_positions(self, x: np.ndarray) -> np.ndarray:
        """Extracts (22, 3) joint positions from state vector."""
        p = np.zeros((self.num_joints, 3), dtype=np.float64)
        for j in range(self.num_joints):
            p[j] = x[j*self.dpj:j*self.dpj+3]
        return p

    def extract_velocities(self, x: np.ndarray) -> np.ndarray:
        """Extracts (22, 3) joint velocities from state vector."""
        v = np.zeros((self.num_joints, 3), dtype=np.float64)
        for j in range(self.num_joints):
            v[j] = x[j*self.dpj+3:j*self.dpj+6]
        return v


class ExtendedKalmanFilter:
    """EKF fusing 3D joint positions and non-linear radar radial velocity measurements."""
    def __init__(self, motion_model: KalmanMotionModel, Q: np.ndarray, R_pos: float = 0.05, R_rad: float = 0.10):
        self.motion = motion_model
        self.Q = Q.copy()
        self.state_dim = motion_model.state_dim
        self.num_joints = motion_model.num_joints
        self.dpj = motion_model.dim_per_joint
        self.linear_kf = LinearKalmanFilter(motion_model, Q, R_pos)
        self.radar_model = RadarMeasurementModel(sigma_pos=R_pos, sigma_rad=R_rad)
        self.R_rad = R_rad ** 2

    def update_with_radial_velocity(
        self, x_pred: np.ndarray, P_pred: np.ndarray,
        y_pos: Optional[np.ndarray], v_r_meas: Optional[float], mask: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Performs joint EKF update fusing 3D position and radial velocity."""
        if mask == 0:
            return x_pred.copy(), P_pred.copy(), 0.0

        x_curr = x_pred.copy()
        P_curr = P_pred.copy()

        # Step 1: Linear update on 3D positions if available
        if y_pos is not None:
            x_curr, P_curr = self.linear_kf.update(x_curr, P_curr, y_pos, mask=1)

        # Step 2: Non-linear EKF update on target root radial velocity
        radial_residual = 0.0
        if v_r_meas is not None:
            p_root = x_curr[0:3]
            v_root = x_curr[3:6]
            v_r_pred = self.radar_model.compute_radial_velocity(p_root, v_root)
            radial_residual = float(v_r_meas - v_r_pred)

            H_p, H_v = self.radar_model.compute_radial_velocity_jacobian(p_root, v_root)
            H_rad = np.zeros((1, self.state_dim), dtype=np.float64)
            H_rad[0, 0:3] = H_p
            H_rad[0, 3:6] = H_v

            S = float(H_rad @ P_curr @ H_rad.T + self.R_rad)
            K = (P_curr @ H_rad.T) / S # shape (state_dim, 1)

            x_curr = x_curr + np.squeeze(K * radial_residual, axis=-1)
            I_KH = np.eye(self.state_dim, dtype=np.float64) - K @ H_rad
            P_curr = I_KH @ P_curr @ I_KH.T + (K * self.R_rad) @ K.T

        return x_curr, P_curr, radial_residual
