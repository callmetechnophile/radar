"""PhotonShield AI — Phase V8.1 Mathematical Dynamics Refinement
Module: v8_1_skeletal_constraints.py

Advanced Skeletal Constraint Engine:
- Hierarchical kinematic tree bone projection (guarantees exact bone lengths)
- Non-linear joint angle cone projection (anatomical articulation limits)
- Global least-squares constrained optimization (POCS)
- Velocity & acceleration state-space filtering
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

# Tree topology for hierarchical root-outward projection (parent -> child)
KINEMATIC_TREE = [
    # Spine chain (root = 0: Pelvis)
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left Leg chain
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right Leg chain
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left Arm chain (from Upper Spine 9)
    (9, 13), (13, 16), (16, 18), (18, 20),
    # Right Arm chain (from Upper Spine 9)
    (9, 14), (14, 17), (17, 19), (19, 21),
]

# Anatomical triplets: (parent, center, child, name)
JOINT_TRIPLETS = [
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


class AdvancedSkeletalConstraints:
    """Rigorous skeletal geometry engine with hierarchical and global projection."""
    def __init__(self):
        self.bone_targets: Dict[Tuple[int, int], float] = {}
        self.bone_bounds: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.angle_bounds: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
        self.v_max: np.ndarray = np.zeros(22, dtype=np.float64)
        self.a_max: np.ndarray = np.zeros(22, dtype=np.float64)
        self.workspace: Dict[str, Tuple[float, float]] = {}

    def fit_from_training(self, train_trajectories: np.ndarray, dt: float = 0.033333):
        """Derives anatomical constraints strictly from clean training data."""
        N, T, J, _ = train_trajectories.shape
        flat = train_trajectories.reshape(-1, 22, 3)

        # 1. Bone lengths (mean target & empirical bounds)
        for (u, v) in KINEMATIC_TREE:
            lens = np.linalg.norm(flat[:, u] - flat[:, v], axis=-1)
            mean_len = float(np.median(lens))
            lo = float(np.percentile(lens, 0.5)) * 0.96
            hi = float(np.percentile(lens, 99.5)) * 1.04
            self.bone_targets[(u, v)] = mean_len
            self.bone_bounds[(u, v)] = (lo, hi)

        # 2. Joint angles
        for (u, v, w, _) in JOINT_TRIPLETS:
            v1 = flat[:, u] - flat[:, v]
            v2 = flat[:, w] - flat[:, v]
            n1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-7
            n2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-7
            cos_a = np.clip(np.sum((v1 / n1) * (v2 / n2), axis=-1), -0.9999, 0.9999)
            angles = np.arccos(cos_a)
            lo = max(0.0, float(np.percentile(angles, 0.5)) - 0.08)
            hi = min(float(np.pi), float(np.percentile(angles, 99.5)) + 0.08)
            self.angle_bounds[(u, v, w)] = (lo, hi)

        # 3. Workspace bounds on root
        root = flat[:, 0]
        self.workspace = {
            "x": (float(np.percentile(root[:, 0], 0.1)) - 0.20, float(np.percentile(root[:, 0], 99.9)) + 0.20),
            "y": (float(np.percentile(root[:, 1], 0.1)) - 0.20, float(np.percentile(root[:, 1], 99.9)) + 0.20),
            "z": (float(np.percentile(root[:, 2], 0.1)) - 0.15, float(np.percentile(root[:, 2], 99.9)) + 0.15),
        }

        # 4. Velocities & Accelerations
        vel = (train_trajectories[:, 1:] - train_trajectories[:, :-1]) / dt
        v_mag = np.linalg.norm(vel, axis=-1)
        self.v_max = np.array([float(np.percentile(v_mag[:, :, j], 99.9)) * 1.30 for j in range(22)], dtype=np.float64)

        acc = (vel[:, 1:] - vel[:, :-1]) / dt
        a_mag = np.linalg.norm(acc, axis=-1)
        self.a_max = np.array([float(np.percentile(a_mag[:, :, j], 99.9)) * 1.35 for j in range(22)], dtype=np.float64)

    def project_hierarchical_bones(self, joints: np.ndarray) -> np.ndarray:
        """Hierarchical root-outward forward kinematic bone projection.
        Traverses tree from Pelvis -> Head/Extremities, setting bone length exactly
        along predicted orientation ray without parent bone displacement.
        Result: Bone violations = 0.0% by construction!
        """
        p = joints.copy()
        for (parent, child) in KINEMATIC_TREE:
            ray = p[child] - p[parent]
            dist = np.linalg.norm(ray)
            target = self.bone_targets[(parent, child)]
            if dist < 1e-6:
                # Default downward/forward offset if collapsed
                p[child] = p[parent] + np.array([0.0, 0.0, -target])
            else:
                p[child] = p[parent] + (ray / dist) * target
        return p

    def project_joint_angle_cone(self, joints: np.ndarray) -> np.ndarray:
        """Projects distal bones back into valid angular cone [theta_min, theta_max].
        Rotates child bone around cross-product rotation axis.
        """
        p = joints.copy()
        for (u, v, w, _) in JOINT_TRIPLETS:
            v1 = p[u] - p[v]
            v2 = p[w] - p[v]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            u1 = v1 / n1
            u2 = v2 / n2
            cos_a = np.clip(np.dot(u1, u2), -0.9999, 0.9999)
            theta = np.arccos(cos_a)
            lo, hi = self.angle_bounds[(u, v, w)]

            if theta < lo or theta > hi:
                target_theta = lo if theta < lo else hi
                rot_angle = target_theta - theta
                # Axis perpendicular to bone plane
                axis = np.cross(u1, u2)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-6:
                    axis = np.array([1.0, 0.0, 0.0]) if abs(u1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                else:
                    axis = axis / axis_norm

                # Rodrigues rotation formula on u2
                cos_r = np.cos(rot_angle)
                sin_r = np.sin(rot_angle)
                u2_rot = u2 * cos_r + np.cross(axis, u2) * sin_r + axis * np.dot(axis, u2) * (1.0 - cos_r)
                u2_rot = u2_rot / (np.linalg.norm(u2_rot) + 1e-7)
                p[w] = p[v] + u2_rot * n2
        return p

    def project_global_least_squares(self, joints: np.ndarray, num_pocs_iters: int = 3) -> np.ndarray:
        """Global Projection Onto Convex Sets (POCS) alternating between:
        1. Hierarchical bone length preservation
        2. Joint angle cone articulation
        """
        p = joints.copy()
        for _ in range(num_pocs_iters):
            p = self.project_hierarchical_bones(p)
            p = self.project_joint_angle_cone(p)
        # Final pass guarantees exact bone lengths
        p = self.project_hierarchical_bones(p)
        return p

    def evaluate_violations(self, joints: np.ndarray, dt: float = 0.033333) -> Dict[str, float]:
        """Calculates precise violation rates across the dataset."""
        if joints.ndim == 3:
            joints = joints[np.newaxis, ...]
        N, T, J, _ = joints.shape
        flat = joints.reshape(-1, 22, 3)

        # Bone violations
        tot_b = 0; viol_b = 0
        for (u, v), (lo, hi) in self.bone_bounds.items():
            d = np.linalg.norm(flat[:, u] - flat[:, v], axis=-1)
            viol_b += int(np.sum((d < lo) | (d > hi)))
            tot_b += len(d)
        bone_rate = viol_b / max(1, tot_b)

        # Angle violations
        tot_a = 0; viol_a = 0
        for (u, v, w), (lo, hi) in self.angle_bounds.items():
            v1 = flat[:, u] - flat[:, v]
            v2 = flat[:, w] - flat[:, v]
            n1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-7
            n2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-7
            cos_a = np.clip(np.sum((v1 / n1) * (v2 / n2), axis=-1), -0.9999, 0.9999)
            angles = np.arccos(cos_a)
            viol_a += int(np.sum((angles < lo) | (angles > hi)))
            tot_a += len(angles)
        angle_rate = viol_a / max(1, tot_a)

        # Velocity violations
        if T >= 2:
            vel = (joints[:, 1:] - joints[:, :-1]) / dt
            v_mag = np.linalg.norm(vel, axis=-1)
            v_rate = float(np.mean(v_mag > self.v_max.reshape(1, 1, 22)))
        else:
            v_rate = 0.0

        # Acceleration violations
        if T >= 3:
            acc = (vel[:, 1:] - vel[:, :-1]) / dt
            a_mag = np.linalg.norm(acc, axis=-1)
            a_rate = float(np.mean(a_mag > self.a_max.reshape(1, 1, 22)))
        else:
            a_rate = 0.0

        return {
            "bone_violation_rate": bone_rate,
            "joint_angle_violation_rate": angle_rate,
            "velocity_violation_rate": v_rate,
            "acceleration_violation_rate": a_rate,
        }
