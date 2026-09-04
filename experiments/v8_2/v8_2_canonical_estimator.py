"""PhotonShield AI — Phase V8.2 Mathematical Stress Validation
Module: v8_2_canonical_estimator.py

Encapsulates the exact canonical V8.1 analytical estimator:
- Jerk-Damped Kinematic Model (gamma = 0.96)
- Adaptive Process Noise Q_t
- Dynamic Observation Confidence c_t in [0, 1]
- Non-linear Radar Radial Velocity EKF with confidence weighting
- Multi-Rate Geometric Sequential LiDAR Kalman Fusion
- Hierarchical Forward Kinematic Tree + Global POCS Joint Angle Cones
- Velocity-Damped Extrapolation & Covariance Exponential Inflation

DO NOT modify this estimator or tune parameters per stress test.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from experiments.v8_1.v8_1_dynamics_models import DynamicalMotionModels
from experiments.v8_1.v8_1_skeletal_constraints import AdvancedSkeletalConstraints
from experiments.v8_1.v8_1_sensor_fusion import SensorFusionConfidenceEngine
from experiments.v8_math.v8_math_measurements import RadarMeasurementModel


class CanonicalV81Estimator:
    """Rigorous implementation of the frozen V8.1 canonical mathematical estimator."""
    def __init__(
        self,
        constraints: AdvancedSkeletalConstraints,
        dt: float = 0.033333,
        num_joints: int = 22,
        damping: float = 0.96,
        sigma_a: float = 2.0,
        sigma_radar_pos: float = 0.05,
        sigma_lidar_pos: float = 0.02,
        sigma_rad_vel: float = 0.10,
    ):
        self.dt = float(dt)
        self.num_joints = num_joints
        self.constraints = constraints
        self.damping = damping
        self.sigma_a = sigma_a
        self.sigma_radar_pos = sigma_radar_pos
        self.sigma_lidar_pos = sigma_lidar_pos
        self.sigma_rad_vel = sigma_rad_vel

        # Dynamics factory & fusion engine
        self.dyn_factory = DynamicalMotionModels(dt=self.dt, num_joints=self.num_joints)
        self.fusion_engine = SensorFusionConfidenceEngine(
            sigma_pos_base=sigma_radar_pos,
            sigma_rad_base=sigma_rad_vel,
            sigma_lidar_base=sigma_lidar_pos,
        )

        # Canonical Model: Jerk-Damped Kinematics (state dim = 6 per joint)
        self.F, self.base_Q, self.dpj = self.dyn_factory.get_jerk_limited_model(
            damping=self.damping, sigma_a=self.sigma_a
        )
        self.state_dim = self.num_joints * self.dpj # 132

        # Standard observation matrix for position (66 x 132)
        self.H_pos = np.zeros((self.num_joints * 3, self.state_dim), dtype=np.float64)
        for j in range(self.num_joints):
            self.H_pos[j*3:(j+1)*3, j*self.dpj:j*self.dpj+3] = np.eye(3, dtype=np.float64)

    def initialize(self, init_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initializes state vector and covariance matrix from first observation."""
        x = np.zeros(self.state_dim, dtype=np.float64)
        for j in range(self.num_joints):
            x[j*self.dpj:j*self.dpj+3] = init_pos[j]
        P = np.eye(self.state_dim, dtype=np.float64) * 0.01
        return x, P

    def filter_sequence(
        self,
        radar_meas: Optional[np.ndarray], # [T, 22, 3] or None
        radar_mask: np.ndarray,          # [T] binary
        radar_v_r: Optional[np.ndarray],  # [T] radial velocities or None
        lidar_meas: Optional[np.ndarray], # [T, 22, 3] or None
        lidar_mask: np.ndarray,          # [T] binary
        dt_seq: Optional[np.ndarray] = None, # [T] time intervals (default self.dt)
        calib_offset: Optional[np.ndarray] = None, # [3] translation error
        calib_rot: Optional[np.ndarray] = None,    # [3, 3] rotation error
        ablation_flags: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Executes full canonical V8.1 tracking across sequence T."""
        if ablation_flags is None:
            ablation_flags = {}

        use_jerk_damping = ablation_flags.get("use_jerk_damping", True)
        use_adaptive_q   = ablation_flags.get("use_adaptive_q", True)
        use_confidence_gating = ablation_flags.get("use_confidence_gating", True)
        use_radial_ekf   = ablation_flags.get("use_radial_ekf", True)
        use_lidar        = ablation_flags.get("use_lidar", True)
        use_constraints  = ablation_flags.get("use_constraints", True)
        use_cov_inflation= ablation_flags.get("use_cov_inflation", True)

        T = len(radar_mask)
        # Select transition matrix
        if use_jerk_damping:
            F_curr = self.F
        else:
            F_curr, _, _ = self.dyn_factory.get_cv_model(sigma_a=self.sigma_a)

        # Initial measurement
        first_obs = None
        for t in range(T):
            if radar_mask[t] == 1 and radar_meas is not None:
                first_obs = radar_meas[t]
                break
            elif use_lidar and lidar_mask[t] == 1 and lidar_meas is not None:
                first_obs = lidar_meas[t]
                break
        if first_obs is None:
            first_obs = np.zeros((self.num_joints, 3))

        x, P = self.initialize(first_obs)

        est_positions = []
        est_velocities = []
        uncertainty_traces = []
        confidences = []
        rad_residuals = []

        consec_missing = 0
        last_e_r = 0.0

        for t in range(T):
            m_r = int(radar_mask[t])
            m_l = int(lidar_mask[t]) if use_lidar else 0

            # Step dt
            step_dt = float(dt_seq[t]) if dt_seq is not None else self.dt

            # 1. Observation Confidence c_t
            if use_confidence_gating:
                c_t = self.fusion_engine.compute_observation_confidence(
                    radar_mask=m_r,
                    lidar_mask=m_l,
                    radial_residual=last_e_r,
                    consecutive_missing_frames=consec_missing,
                )
            else:
                c_t = 1.0 if (m_r == 1 or m_l == 1) else 0.1
            confidences.append(c_t)

            # 2. Adaptive Process Noise Q_t
            if use_adaptive_q:
                # Extract current velocity magnitude
                vel_mags = [np.linalg.norm(x[j*self.dpj+3:j*self.dpj+6]) for j in range(self.num_joints)]
                v_mean = float(np.mean(vel_mags))
                innov_mag = abs(last_e_r)
                Q_t = self.dyn_factory.compute_adaptive_q(
                    self.base_Q, v_mean, innov_mag, consec_missing, c_t
                )
            else:
                Q_t = self.base_Q

            # 3. Kalman Prediction Step
            x_pred = F_curr @ x
            P_pred = F_curr @ P @ F_curr.T + Q_t

            if not use_cov_inflation and (m_r == 0 and m_l == 0):
                # suppress covariance growth if inflation is ablated
                P_pred = P.copy()

            # 4. Measurement Update Step
            if m_r == 1 or m_l == 1:
                consec_missing = 0
                # Synthesize / combine observations
                # Adaptive R based on confidence
                R_rad_val = (self.sigma_radar_pos ** 2) / max(0.05, c_t)
                R_lid_val = (self.sigma_lidar_pos ** 2) / max(0.05, c_t)

                # Optimal Multi-Modal Geometric Sensor Fusion
                if use_lidar and m_l == 1 and lidar_meas is not None and m_r == 1 and radar_meas is not None:
                    y_l = lidar_meas[t].copy()
                    if calib_offset is not None:
                        y_l = y_l + calib_offset
                    if calib_rot is not None:
                        y_l = (calib_rot @ y_l.T).T
                    y_r = radar_meas[t]

                    w_r = 1.0 / R_rad_val
                    w_l = 1.0 / R_lid_val
                    y_fused = (w_r * y_r + w_l * y_l) / (w_r + w_l)
                    R_fused_val = 1.0 / (w_r + w_l)

                    y_flat = y_fused.ravel()
                    R_mat = R_fused_val * np.eye(len(y_flat), dtype=np.float64)
                    innov = y_flat - self.H_pos @ x_pred
                    S = self.H_pos @ P_pred @ self.H_pos.T + R_mat
                    K = P_pred @ self.H_pos.T @ np.linalg.inv(S)
                    x_pred = x_pred + K @ innov
                    I_KH = np.eye(self.state_dim) - K @ self.H_pos
                    P_pred = I_KH @ P_pred @ I_KH.T + K @ R_mat @ K.T

                elif use_lidar and m_l == 1 and lidar_meas is not None:
                    y_l = lidar_meas[t].copy()
                    if calib_offset is not None:
                        y_l = y_l + calib_offset
                    if calib_rot is not None:
                        y_l = (calib_rot @ y_l.T).T
                    y_flat = y_l.ravel()
                    R_mat = R_lid_val * np.eye(len(y_flat), dtype=np.float64)
                    innov = y_flat - self.H_pos @ x_pred
                    S = self.H_pos @ P_pred @ self.H_pos.T + R_mat
                    K = P_pred @ self.H_pos.T @ np.linalg.inv(S)
                    x_pred = x_pred + K @ innov
                    I_KH = np.eye(self.state_dim) - K @ self.H_pos
                    P_pred = I_KH @ P_pred @ I_KH.T + K @ R_mat @ K.T

                elif m_r == 1 and radar_meas is not None:
                    y_flat = radar_meas[t].ravel()
                    R_mat = R_rad_val * np.eye(len(y_flat), dtype=np.float64)
                    innov = y_flat - self.H_pos @ x_pred
                    S = self.H_pos @ P_pred @ self.H_pos.T + R_mat
                    K = P_pred @ self.H_pos.T @ np.linalg.inv(S)
                    x_pred = x_pred + K @ innov
                    I_KH = np.eye(self.state_dim) - K @ self.H_pos
                    P_pred = I_KH @ P_pred @ I_KH.T + K @ R_mat @ K.T

                # Update with Radar Radial Velocity (EKF)
                if use_radial_ekf and m_r == 1 and radar_v_r is not None:
                    p_root = x_pred[0:3]
                    v_root = x_pred[3:6]
                    v_r_meas_val = float(radar_v_r[t])
                    v_r_pred = RadarMeasurementModel.compute_radial_velocity(p_root, v_root)
                    e_r = v_r_meas_val - v_r_pred
                    last_e_r = e_r
                    rad_residuals.append(e_r)

                    # Jacobian of v_r w.r.t root joint
                    H_p, H_v = RadarMeasurementModel.compute_radial_velocity_jacobian(p_root, v_root)
                    H_rad = np.zeros((1, self.state_dim), dtype=np.float64)
                    H_rad[0, 0:3] = H_p
                    H_rad[0, 3:6] = H_v

                    R_rad = np.array([[(self.sigma_rad_vel ** 2) / max(0.05, c_t)]], dtype=np.float64)
                    S_rad = H_rad @ P_pred @ H_rad.T + R_rad
                    K_rad = (P_pred @ H_rad.T) / S_rad[0, 0]
                    x_pred = x_pred + K_rad.ravel() * e_r
                    I_KH_rad = np.eye(self.state_dim) - K_rad @ H_rad
                    P_pred = I_KH_rad @ P_pred @ I_KH_rad.T + K_rad @ R_rad @ K_rad.T
                else:
                    last_e_r = 0.0
                    rad_residuals.append(0.0)

                x_up, P_up = x_pred, P_pred
            else:
                consec_missing += 1
                last_e_r = 0.0
                rad_residuals.append(0.0)
                x_up, P_up = x_pred, P_pred

            # 5. Skeletal Constraint Projection (POCS)
            p_t = np.array([x_up[j*self.dpj:j*self.dpj+3] for j in range(self.num_joints)])
            if use_constraints:
                p_t = self.constraints.project_global_least_squares(p_t, num_pocs_iters=3)
                for j in range(self.num_joints):
                    x_up[j*self.dpj:j*self.dpj+3] = p_t[j]

            v_t = np.array([x_up[j*self.dpj+3:j*self.dpj+6] for j in range(self.num_joints)])

            x, P = x_up, P_up
            est_positions.append(p_t)
            est_velocities.append(v_t)
            uncertainty_traces.append(float(np.trace(P)))

        return {
            "pred_positions": np.stack(est_positions, axis=0),   # [T, 22, 3]
            "pred_velocities": np.stack(est_velocities, axis=0), # [T, 22, 3]
            "uncertainty_trace": np.array(uncertainty_traces),   # [T]
            "confidences": np.array(confidences),                # [T]
            "radial_residuals": np.array(rad_residuals),         # [T]
        }
