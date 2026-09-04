"""PhotonShield AI — V8-MATH Mathematical State Estimation Foundation
Module: v8_math_constraints.py

Skeletal, kinematic, and dynamic physical constraints derived strictly from TRAINING data.
Includes unconstrained, soft penalty, hard clipping, and projection-based constrained estimation.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

# Canonical M4Human 22-joint topology
BONE_PAIRS = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),       # Spine
    (0, 1), (1, 4), (4, 7), (7, 10),                 # Left Leg
    (0, 2), (2, 5), (5, 8), (8, 11),                 # Right Leg
    (9, 13), (13, 16), (16, 18), (18, 20),           # Left Arm
    (9, 14), (14, 17), (17, 19), (19, 21),           # Right Arm
]

ANGLE_TRIPLETS = [
    (1, 4, 7, "L Knee"),
    (2, 5, 8, "R Knee"),
    (4, 7, 10, "L Ankle"),
    (5, 8, 11, "R Ankle"),
    (0, 1, 4, "L Hip"),
    (0, 2, 5, "R Hip"),
    (16, 18, 20, "L Elbow"),
    (17, 19, 21, "R Elbow"),
    (13, 16, 18, "L Shoulder"),
    (14, 17, 19, "R Shoulder"),
    (0, 3, 6, "Spine Low"),
    (3, 6, 9, "Spine Mid"),
    (6, 9, 12, "Spine Up"),
    (9, 12, 15, "Neck/Head"),
]


class HumanPhysicalConstraints:
    """Manages physical human constraints estimated strictly from TRAINING data."""
    def __init__(self):
        self.bone_bounds: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.angle_bounds: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
        self.v_max: np.ndarray = np.zeros(22, dtype=np.float64)
        self.a_max: np.ndarray = np.zeros(22, dtype=np.float64)
        self.workspace_bounds: Dict[str, Tuple[float, float]] = {}

    def fit_from_training_trajectories(self, trajectories: np.ndarray, dt: float = 0.033333):
        """Fits physical boundaries from clean training data only.
        trajectories: shape [N_seq, T, 22, 3] in meters.
        """
        N, T, J, _ = trajectories.shape
        # 1. Bone lengths
        for (u, v) in BONE_PAIRS:
            lens = np.linalg.norm(trajectories[:, :, u] - trajectories[:, :, v], axis=-1).ravel()
            lo = float(np.percentile(lens, 0.1)) * 0.97
            hi = float(np.percentile(lens, 99.9)) * 1.03
            self.bone_bounds[(u, v)] = (lo, hi)

        # 2. Joint angles
        for (u, v, w, _) in ANGLE_TRIPLETS:
            v1 = trajectories[:, :, u] - trajectories[:, :, v]
            v2 = trajectories[:, :, w] - trajectories[:, :, v]
            n1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-7
            n2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-7
            cos_a = np.clip(np.sum((v1 / n1) * (v2 / n2), axis=-1), -0.9999, 0.9999)
            angles = np.arccos(cos_a).ravel()
            lo = max(0.0, float(np.percentile(angles, 0.1)) - 0.05)
            hi = min(float(np.pi), float(np.percentile(angles, 99.9)) + 0.05)
            self.angle_bounds[(u, v, w)] = (lo, hi)

        # 3. Workspace bounds on root
        root = trajectories[:, :, 0, :].reshape(-1, 3)
        self.workspace_bounds = {
            "x": (float(np.percentile(root[:, 0], 0.1)) - 0.15, float(np.percentile(root[:, 0], 99.9)) + 0.15),
            "y": (float(np.percentile(root[:, 1], 0.1)) - 0.15, float(np.percentile(root[:, 1], 99.9)) + 0.15),
            "z": (float(np.percentile(root[:, 2], 0.1)) - 0.10, float(np.percentile(root[:, 2], 99.9)) + 0.10),
        }

        # 4. Velocities
        vel = (trajectories[:, 1:] - trajectories[:, :-1]) / dt # [N, T-1, 22, 3]
        v_mag = np.linalg.norm(vel, axis=-1) # [N, T-1, 22]
        self.v_max = np.array([float(np.percentile(v_mag[:, :, j], 99.9)) * 1.25 for j in range(22)], dtype=np.float64)

        # 5. Accelerations
        acc = (vel[:, 1:] - vel[:, :-1]) / dt # [N, T-2, 22, 3]
        a_mag = np.linalg.norm(acc, axis=-1) # [N, T-2, 22]
        # Robust acceleration limits with smoothing sensitivity analysis
        self.a_max = np.array([float(np.percentile(a_mag[:, :, j], 99.9)) * 1.30 for j in range(22)], dtype=np.float64)

    def evaluate_violations(self, joints: np.ndarray, dt: float = 0.033333) -> Dict[str, float]:
        """Calculates violation rates for bones, angles, velocities, accelerations, and workspace.
        joints: shape [T, 22, 3] or [N, T, 22, 3]
        """
        if joints.ndim == 3:
            joints = joints[np.newaxis, ...]
        N, T, J, _ = joints.shape
        flat = joints.reshape(-1, 22, 3)

        # Bone violations
        total_bones = 0
        viol_bones = 0
        for (u, v), (lo, hi) in self.bone_bounds.items():
            d = np.linalg.norm(flat[:, u] - flat[:, v], axis=-1)
            viol_bones += int(np.sum((d < lo) | (d > hi)))
            total_bones += len(d)
        bone_viol_rate = viol_bones / max(1, total_bones)

        # Joint angle violations
        total_angles = 0
        viol_angles = 0
        for (u, v, w), (lo, hi) in self.angle_bounds.items():
            v1 = flat[:, u] - flat[:, v]
            v2 = flat[:, w] - flat[:, v]
            n1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-7
            n2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-7
            cos_a = np.clip(np.sum((v1 / n1) * (v2 / n2), axis=-1), -0.9999, 0.9999)
            angles = np.arccos(cos_a)
            viol_angles += int(np.sum((angles < lo) | (angles > hi)))
            total_angles += len(angles)
        angle_viol_rate = viol_angles / max(1, total_angles)

        # Workspace violations
        root = flat[:, 0]
        ws = self.workspace_bounds
        ws_viol = (root[:, 0] < ws["x"][0]) | (root[:, 0] > ws["x"][1]) | \
                  (root[:, 1] < ws["y"][0]) | (root[:, 1] > ws["y"][1]) | \
                  (root[:, 2] < ws["z"][0]) | (root[:, 2] > ws["z"][1])
        ws_viol_rate = float(np.mean(ws_viol))

        # Velocity violations
        if T >= 2:
            vel = (joints[:, 1:] - joints[:, :-1]) / dt
            v_mag = np.linalg.norm(vel, axis=-1)
            v_viol_rate = float(np.mean(v_mag > self.v_max.reshape(1, 1, 22)))
        else:
            v_viol_rate = 0.0

        # Acceleration violations
        if T >= 3:
            acc = (vel[:, 1:] - vel[:, :-1]) / dt
            a_mag = np.linalg.norm(acc, axis=-1)
            a_viol_rate = float(np.mean(a_mag > self.a_max.reshape(1, 1, 22)))
        else:
            a_viol_rate = 0.0

        return {
            "bone_violation_rate": bone_viol_rate,
            "joint_angle_violation_rate": angle_viol_rate,
            "workspace_violation_rate": ws_viol_rate,
            "velocity_violation_rate": v_viol_rate,
            "acceleration_violation_rate": a_viol_rate,
        }

    def project_bone_constraints(self, joints: np.ndarray, num_iterations: int = 5) -> np.ndarray:
        """Projects joint positions onto valid bone-length constraints via iterative constraint projection.
        joints: shape (22, 3)
        """
        p = joints.copy()
        for _ in range(num_iterations):
            for (u, v), (lo, hi) in self.bone_bounds.items():
                delta = p[v] - p[u]
                d = np.linalg.norm(delta)
                if d < 1e-6:
                    continue
                if d < lo:
                    corr = 0.5 * (lo - d) * (delta / d)
                    if u != 0: p[u] -= corr
                    p[v] += corr
                elif d > hi:
                    corr = 0.5 * (d - hi) * (delta / d)
                    if u != 0: p[u] += corr
                    p[v] -= corr
        return p

    def project_velocity_constraints(self, p_curr: np.ndarray, p_prev: np.ndarray, dt: float = 0.033333) -> np.ndarray:
        """Clamps displacement to maximum velocity bounds."""
        p_proj = p_curr.copy()
        v = (p_curr - p_prev) / dt
        for j in range(22):
            v_mag = np.linalg.norm(v[j])
            if v_mag > self.v_max[j]:
                v_clamped = (self.v_max[j] / v_mag) * v[j]
                p_proj[j] = p_prev[j] + v_clamped * dt
        return p_proj
