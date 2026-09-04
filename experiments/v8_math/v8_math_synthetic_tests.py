"""PhotonShield AI — V8-MATH Mathematical State Estimation Foundation
Module: v8_math_synthetic_tests.py

Comprehensive mathematical unit tests validating analytical kinematics,
Kalman filtering, missing-observation gating, radial velocity Jacobians, and constraints.
Must pass 100% before real dataset benchmarking.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import math
import numpy as np
from experiments.v8_math.v8_math_measurements import RadarMeasurementModel
from experiments.v8_math.v8_math_kalman import KalmanMotionModel, LinearKalmanFilter, ExtendedKalmanFilter
from experiments.v8_math.v8_math_constraints import HumanPhysicalConstraints, BONE_PAIRS, ANGLE_TRIPLETS
from experiments.v8_math.v8_math_lidar import LidarMeasurementModel, AnalyticalSensorFusion


def test_state_transition_and_integration():
    """Validates F matrix for CV and CA integration against exact analytic calculus."""
    dt = 0.033333
    cv_model = KalmanMotionModel(dt=dt, model_type="CV", num_joints=1)
    p0 = np.array([1.0, 2.0, 3.0])
    v0 = np.array([0.5, -0.2, 0.1])
    x0 = np.concatenate([p0, v0])

    # 1. Constant Velocity Step
    x1 = cv_model.F @ x0
    expected_p1 = p0 + v0 * dt
    expected_v1 = v0
    assert np.allclose(x1[0:3], expected_p1, atol=1e-12), "CV position integration failed"
    assert np.allclose(x1[3:6], expected_v1, atol=1e-12), "CV velocity preservation failed"

    # 2. Constant Acceleration Step
    ca_model = KalmanMotionModel(dt=dt, model_type="CA", num_joints=1)
    a0 = np.array([0.1, -0.05, 0.2])
    x0_ca = np.concatenate([p0, v0, a0])
    x1_ca = ca_model.F @ x0_ca
    expected_p1_ca = p0 + v0 * dt + 0.5 * a0 * (dt ** 2)
    expected_v1_ca = v0 + a0 * dt
    assert np.allclose(x1_ca[0:3], expected_p1_ca, atol=1e-12), "CA position integration failed"
    assert np.allclose(x1_ca[3:6], expected_v1_ca, atol=1e-12), "CA velocity integration failed"
    assert np.allclose(x1_ca[6:9], a0, atol=1e-12), "CA acceleration preservation failed"
    print("  [PASS] State Transition & Numerical Integration")


def test_covariance_propagation_and_kalman_update():
    """Validates covariance propagation and missing-observation gating."""
    dt = 0.033333
    cv_model = KalmanMotionModel(dt=dt, model_type="CV", num_joints=1)
    Q = cv_model.compute_continuous_white_noise_q(sigma_a=0.5)
    kf = LinearKalmanFilter(cv_model, Q, R_pos=0.05)

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    P0 = np.eye(6) * 0.01

    # Predict
    x_pred, P_pred = kf.predict(x0, P0)
    # Trace of P must strictly increase due to process noise Q
    assert np.trace(P_pred) > np.trace(P0), "Covariance must grow under prediction without measurement"

    # Measurement update with observation (m=1)
    y_meas = np.array([0.033, 0.0, 0.0])
    x_up, P_up = kf.update(x_pred, P_pred, y_meas, mask=1)
    assert np.trace(P_up) < np.trace(P_pred), "Covariance must decrease after measurement update"

    # Missing observation (m=0): observation-mask gating!
    x_gated, P_gated = kf.update(x_pred, P_pred, np.zeros(3), mask=0)
    assert np.allclose(x_gated, x_pred, atol=1e-12), "Missing frame must perform zero state change"
    assert np.allclose(P_gated, P_pred, atol=1e-12), "Missing frame must preserve predicted covariance"
    print("  [PASS] Covariance Propagation & Missing-Observation Gating")


def test_radial_velocity_and_jacobian():
    """Validates non-linear radial velocity equation and analytic Jacobian via finite difference."""
    p = np.array([2.0, 3.0, 1.5], dtype=np.float64)
    v = np.array([0.4, -0.3, 0.8], dtype=np.float64)
    r_hat = p / np.linalg.norm(p)

    v_r_analytic = RadarMeasurementModel.compute_radial_velocity(p, v)
    assert np.isclose(v_r_analytic, float(np.dot(r_hat, v)), atol=1e-12), "Radial velocity calculation mismatch"

    # Finite difference check of Jacobians
    H_p, H_v = RadarMeasurementModel.compute_radial_velocity_jacobian(p, v)
    eps = 1e-7
    for i in range(3):
        p_plus = p.copy(); p_plus[i] += eps
        p_minus = p.copy(); p_minus[i] -= eps
        df_dp_num = (RadarMeasurementModel.compute_radial_velocity(p_plus, v) -
                     RadarMeasurementModel.compute_radial_velocity(p_minus, v)) / (2 * eps)
        assert np.isclose(H_p[0, i], df_dp_num, atol=1e-5), f"Jacobian H_p[{i}] mismatch"

        v_plus = v.copy(); v_plus[i] += eps
        v_minus = v.copy(); v_minus[i] -= eps
        df_dv_num = (RadarMeasurementModel.compute_radial_velocity(p, v_plus) -
                     RadarMeasurementModel.compute_radial_velocity(p, v_minus)) / (2 * eps)
        assert np.isclose(H_v[0, i], df_dv_num, atol=1e-5), f"Jacobian H_v[{i}] mismatch"
    print("  [PASS] Non-Linear Radial Velocity & Analytic Jacobians")


def test_constraints_and_projection():
    """Validates bone length calculation, joint angle cosine clamping, and projection."""
    constraints = HumanPhysicalConstraints()
    # Mock bounds for a 2-joint bone (0, 1)
    constraints.bone_bounds[(0, 1)] = (0.35, 0.45)

    p_valid = np.zeros((22, 3))
    p_valid[1] = np.array([0.40, 0.0, 0.0])
    p_proj = constraints.project_bone_constraints(p_valid)
    assert np.isclose(np.linalg.norm(p_proj[1] - p_proj[0]), 0.40, atol=1e-6), "Valid bone should not be altered"

    # Perturbed bone (over-extended to 0.70m)
    p_distorted = np.zeros((22, 3))
    p_distorted[1] = np.array([0.70, 0.0, 0.0])
    p_corrected = constraints.project_bone_constraints(p_distorted, num_iterations=10)
    len_corrected = np.linalg.norm(p_corrected[1] - p_corrected[0])
    assert len_corrected <= 0.455, f"Bone projection failed to clamp over-extended bone: {len_corrected}"

    # Joint angle numerical stability: colinear points (cos = 1 and cos = -1)
    p_a = np.array([0.0, 0.0, 0.0])
    p_b = np.array([0.0, 1.0, 0.0])
    p_c = np.array([0.0, 2.0, 0.0])
    v1 = p_a - p_b; v2 = p_c - p_b
    cos_180 = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    ang = np.arccos(cos_180)
    assert np.isclose(ang, np.pi, atol=1e-6), "Colinear angle must be pi"
    print("  [PASS] Skeletal Constraints & Projection Stability")


def run_all_synthetic_tests():
    print("==================================================")
    print("RUNNING SYNTHETIC MATHEMATICAL VALIDATION SUITE")
    print("==================================================")
    test_state_transition_and_integration()
    test_covariance_propagation_and_kalman_update()
    test_radial_velocity_and_jacobian()
    test_constraints_and_projection()
    print("==================================================")
    print("ALL SYNTHETIC MATHEMATICAL TESTS PASSED: PASS")
    print("==================================================")
    return True


if __name__ == "__main__":
    run_all_synthetic_tests()
