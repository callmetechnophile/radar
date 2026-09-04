"""PhotonShield AI — V8-MATH Mathematical State Estimation Foundation
Module: v8_math_state_estimator.py

Complete mathematical estimator suite:
- M0: Last Observation Hold (ZOH)
- M1: Constant Velocity Extrapolator (CV)
- M2: Constant Acceleration Extrapolator (CA)
- M3: Linear Kalman Filter (KF) with Observation-Mask Gating
- M4: Extended Kalman Filter (EKF) with Radial Velocity
- M5: Constrained State Estimator (with Bone & Angle Projections)
- Full: Multi-Modal Constrained EKF + Radial Consistency + LiDAR Fusion
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from experiments.v8_math.v8_math_measurements import RadarMeasurementModel
from experiments.v8_math.v8_math_kalman import KalmanMotionModel, LinearKalmanFilter, ExtendedKalmanFilter
from experiments.v8_math.v8_math_constraints import HumanPhysicalConstraints
from experiments.v8_math.v8_math_lidar import LidarMeasurementModel, AnalyticalSensorFusion


class MathematicalEstimatorSuite:
    """Manages all 6 mathematical estimators for systematic comparison."""
    def __init__(
        self,
        dt: float = 0.033333,
        num_joints: int = 22,
        sigma_a: float = 2.0,
        sigma_pos: float = 0.05,
        sigma_rad: float = 0.10,
        sigma_lidar: float = 0.02,
        constraints: Optional[HumanPhysicalConstraints] = None,
    ):
        self.dt = float(dt)
        self.num_joints = num_joints
        self.sigma_a = float(sigma_a)
        self.sigma_pos = float(sigma_pos)
        self.sigma_rad = float(sigma_rad)
        self.sigma_lidar = float(sigma_lidar)
        self.constraints = constraints if constraints is not None else HumanPhysicalConstraints()

        # Motion models
        self.motion_cv = KalmanMotionModel(dt=self.dt, model_type="CV", num_joints=num_joints)
        self.motion_ca = KalmanMotionModel(dt=self.dt, model_type="CA", num_joints=num_joints)
        self.Q_cv = self.motion_cv.compute_continuous_white_noise_q(sigma_a=self.sigma_a)
        self.Q_ca = self.motion_ca.compute_continuous_white_noise_q(sigma_a=self.sigma_a)

        # Kalman & EKF instances
        self.kf = LinearKalmanFilter(self.motion_cv, self.Q_cv, R_pos=self.sigma_pos)
        self.ekf = ExtendedKalmanFilter(self.motion_cv, self.Q_cv, R_pos=self.sigma_pos, R_rad=self.sigma_rad)
        self.lidar_model = LidarMeasurementModel(sigma_lidar=self.sigma_lidar, num_joints=num_joints, dim_per_joint=6)
        self.sensor_fusion = AnalyticalSensorFusion(self.kf, self.lidar_model)

    # -------------------------------------------------------------------------
    # M0: Last Observation Hold (ZOH)
    # -------------------------------------------------------------------------
    def run_m0_zoh(self, measurements: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """M0: Last Observation Hold.
        measurements: [T, 22, 3], mask: [T]
        """
        T = len(mask)
        est = np.zeros_like(measurements)
        last_valid = measurements[0].copy()

        for t in range(T):
            if mask[t] == 1:
                last_valid = measurements[t].copy()
            est[t] = last_valid.copy()
        return est

    # -------------------------------------------------------------------------
    # M1: Constant Velocity Extrapolator (CV)
    # -------------------------------------------------------------------------
    def run_m1_cv(self, measurements: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """M1: Constant Velocity Extrapolator.
        measurements: [T, 22, 3], mask: [T]
        """
        T = len(mask)
        est = np.zeros_like(measurements)
        vel = np.zeros((self.num_joints, 3), dtype=np.float64)
        last_p = measurements[0].copy()

        for t in range(T):
            if mask[t] == 1:
                if t > 0:
                    # Filtered velocity update
                    vel = 0.7 * vel + 0.3 * ((measurements[t] - last_p) / self.dt)
                last_p = measurements[t].copy()
                est[t] = last_p.copy()
            else:
                last_p = last_p + vel * self.dt
                est[t] = last_p.copy()
        return est

    # -------------------------------------------------------------------------
    # M2: Constant Acceleration Extrapolator (CA)
    # -------------------------------------------------------------------------
    def run_m2_ca(self, measurements: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """M2: Constant Acceleration Extrapolator."""
        T = len(mask)
        est = np.zeros_like(measurements)
        vel = np.zeros((self.num_joints, 3), dtype=np.float64)
        acc = np.zeros((self.num_joints, 3), dtype=np.float64)
        last_p = measurements[0].copy()

        for t in range(T):
            if mask[t] == 1:
                if t > 1:
                    raw_v = (measurements[t] - last_p) / self.dt
                    raw_a = (raw_v - vel) / self.dt
                    vel = 0.7 * vel + 0.3 * raw_v
                    acc = 0.7 * acc + 0.3 * raw_a
                elif t == 1:
                    vel = (measurements[t] - last_p) / self.dt
                last_p = measurements[t].copy()
                est[t] = last_p.copy()
            else:
                last_p = last_p + vel * self.dt + 0.5 * acc * (self.dt ** 2)
                vel = vel + acc * self.dt
                est[t] = last_p.copy()
        return est

    # -------------------------------------------------------------------------
    # M3: Linear Kalman Filter (KF) with Observation Gating
    # -------------------------------------------------------------------------
    def run_m3_kalman(self, measurements: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """M3: Linear Kalman Filter with observation-mask gating.
        Returns estimated joint positions [T, 22, 3] and state covariance trace per timestep.
        """
        T = len(mask)
        x, P = self.kf.initialize(measurements[0])
        est = np.zeros_like(measurements)
        uncertainties = []

        for t in range(T):
            x_pred, P_pred = self.kf.predict(x, P)
            # Gated update: if mask[t] == 0, performs no measurement update!
            x, P = self.kf.update(x_pred, P_pred, measurements[t], mask=mask[t])
            est[t] = self.kf.extract_positions(x)
            uncertainties.append(float(np.trace(P)))
        return est, uncertainties

    # -------------------------------------------------------------------------
    # M4: Extended Kalman Filter (EKF) with Non-Linear Radial Velocity
    # -------------------------------------------------------------------------
    def run_m4_ekf(
        self, measurements: np.ndarray, mask: np.ndarray, radial_vels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[float], List[float]]:
        """M4: EKF fusing 3D joint positions and radar radial velocity.
        Returns positions, uncertainties, and radial velocity residuals.
        """
        T = len(mask)
        x, P = self.ekf.linear_kf.initialize(measurements[0])
        est = np.zeros_like(measurements)
        uncertainties = []
        residuals = []

        for t in range(T):
            x_pred, P_pred = self.ekf.linear_kf.predict(x, P)
            v_r = float(radial_vels[t]) if radial_vels is not None else None
            x, P, e_r = self.ekf.update_with_radial_velocity(
                x_pred, P_pred, y_pos=measurements[t], v_r_meas=v_r, mask=mask[t]
            )
            est[t] = self.ekf.linear_kf.extract_positions(x)
            uncertainties.append(float(np.trace(P)))
            residuals.append(e_r)
        return est, uncertainties, residuals

    # -------------------------------------------------------------------------
    # M5: Constrained State Estimator
    # -------------------------------------------------------------------------
    def run_m5_constrained(
        self, measurements: np.ndarray, mask: np.ndarray,
        constraint_mode: str = "PROJECTION" # "SOFT", "HARD", "PROJECTION"
    ) -> Tuple[np.ndarray, List[float]]:
        """M5: Kalman state estimator constrained by physical bone lengths and velocity limits."""
        T = len(mask)
        x, P = self.kf.initialize(measurements[0])
        est = np.zeros_like(measurements)
        uncertainties = []

        for t in range(T):
            x_pred, P_pred = self.kf.predict(x, P)
            x, P = self.kf.update(x_pred, P_pred, measurements[t], mask=mask[t])
            p_curr = self.kf.extract_positions(x)

            if constraint_mode == "PROJECTION":
                # Bone projection
                p_curr = self.constraints.project_bone_constraints(p_curr, num_iterations=5)
                # Velocity clamping if past frame available
                if t > 0:
                    p_curr = self.constraints.project_velocity_constraints(p_curr, est[t-1], dt=self.dt)
            elif constraint_mode == "HARD":
                p_curr = self.constraints.project_bone_constraints(p_curr, num_iterations=1)

            # Re-inject projected joints into Kalman state vector
            for j in range(self.num_joints):
                x[j*self.kf.dpj:j*self.kf.dpj+3] = p_curr[j]

            est[t] = p_curr.copy()
            uncertainties.append(float(np.trace(P)))
        return est, uncertainties

    # -------------------------------------------------------------------------
    # Full Mathematical Estimator: Multi-Modal Constrained EKF + LiDAR Fusion
    # -------------------------------------------------------------------------
    def run_full_mathematical_estimator(
        self,
        radar_meas: np.ndarray,
        mask: np.ndarray,
        radial_vels: Optional[np.ndarray] = None,
        lidar_meas: Optional[np.ndarray] = None,
        lidar_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
        """Full mathematical state estimator combining:
        1. Discrete-time constant-velocity motion physics
        2. Observation-mask gated Kalman prediction
        3. Non-linear radar radial velocity update (EKF)
        4. Analytical LiDAR geometric fusion
        5. Skeletal bone and dynamic velocity projections
        """
        T = len(mask)
        if lidar_mask is None:
            lidar_mask = mask.copy()
        if lidar_meas is None:
            lidar_meas = radar_meas.copy()

        x, P = self.kf.initialize(radar_meas[0])
        est = np.zeros_like(radar_meas)
        uncertainties = []
        radar_residuals = []
        lidar_residuals = []

        for t in range(T):
            # 1. Kinematic Prediction
            x_pred, P_pred = self.kf.predict(x, P)

            # 2. Sequential Multi-Modal Sensor Fusion (Radar + LiDAR)
            x_fused, P_fused, f_res = self.sensor_fusion.fuse_step(
                x_pred, P_pred,
                radar_meas=radar_meas[t], lidar_meas=lidar_meas[t],
                radar_mask=mask[t], lidar_mask=lidar_mask[t],
                mode="FUSED"
            )
            radar_residuals.append(f_res["radar_pos_residual"])
            lidar_residuals.append(f_res["lidar_pos_residual"])

            # 3. Non-linear Radial Velocity Update
            if radial_vels is not None and mask[t] == 1:
                x_fused, P_fused, r_res = self.ekf.update_with_radial_velocity(
                    x_fused, P_fused, y_pos=None, v_r_meas=float(radial_vels[t]), mask=1
                )

            # 4. Skeletal & Physical Constraint Projection
            p_curr = self.kf.extract_positions(x_fused)
            p_curr = self.constraints.project_bone_constraints(p_curr, num_iterations=5)
            if t > 0:
                p_curr = self.constraints.project_velocity_constraints(p_curr, est[t-1], dt=self.dt)

            for j in range(self.num_joints):
                x_fused[j*self.kf.dpj:j*self.kf.dpj+3] = p_curr[j]

            x, P = x_fused, P_fused
            est[t] = p_curr.copy()
            uncertainties.append(float(np.trace(P)))

        return est, uncertainties, {
            "radar_residuals": radar_residuals,
            "lidar_residuals": lidar_residuals,
        }
