"""PhotonShield AI — Phase V8.2 Mathematical Stress Validation
Module: v8_2_stress_datasets.py

Comprehensive Stress Data Generation & Motion Synthesis:
- Seen vs Unseen anthropometric subject profiles (scale, height, proportions)
- Categorized action motions: Walking, Standing, Turning, Sitting, Rising, Running, Jumping, Fast Directional Changes, Extreme Athletic
- Multi-horizon sequence synthesis up to 64 frames (preserving temporal continuity)
- Systematic sensor degradation: Radar sparsity, Radar noise, Radial velocity bias, LiDAR dropout
- Temporal corruption patterns: Random, Contiguous, Multiple gaps, Leading gap, Trailing gap
- Timing synchronization jitter, sensor calibration misalignment, coordinate perturbation
- Empirical acceleration and jerk dynamic binning
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from experiments.run_v7_1_m4human_pose import DT_M4HUMAN, M4HumanSequenceDataset
from experiments.v8_math.v8_math_measurements import RadarMeasurementModel


class StressDatasetGenerator:
    """Generates rigorous stress testing sequences and sensor corruptions for V8.2."""
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dt = DT_M4HUMAN

    def get_subject_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Defines seen and unseen anthropometric subject profiles."""
        return {
            "seen_subject_canonical": {"scale_legs": 1.0, "scale_arms": 1.0, "scale_torso": 1.0, "seen": True},
            "seen_subject_slight_variation": {"scale_legs": 1.02, "scale_arms": 0.98, "scale_torso": 1.0, "seen": True},
            "unseen_subject_tall": {"scale_legs": 1.15, "scale_arms": 1.12, "scale_torso": 1.10, "seen": False},
            "unseen_subject_short": {"scale_legs": 0.85, "scale_arms": 0.88, "scale_torso": 0.90, "seen": False},
            "unseen_subject_long_legs": {"scale_legs": 1.20, "scale_arms": 0.95, "scale_torso": 0.92, "seen": False},
            "unseen_subject_long_torso": {"scale_legs": 0.90, "scale_arms": 1.05, "scale_torso": 1.18, "seen": False},
            "unseen_subject_asymmetric": {"scale_legs": 1.08, "scale_arms": 0.92, "scale_torso": 1.05, "seen": False},
        }

    def synthesize_action_sequence(
        self,
        action_type: str,
        T: int = 32,
        subject_profile: Optional[Dict[str, Any]] = None,
        seq_seed: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Synthesizes physically realistic 3D human pose trajectory for specific actions."""
        rng = np.random.default_rng(seq_seed)
        times = np.arange(T) * self.dt

        # Base bone offsets
        base_offsets = np.zeros((22, 3), dtype=np.float64)
        base_offsets[1] = [-0.10, 0.0, -0.05]   # L Hip
        base_offsets[2] = [0.10, 0.0, -0.05]    # R Hip
        base_offsets[3] = [0.0, 0.0, 0.15]      # Spine 1
        base_offsets[4] = [-0.10, 0.0, -0.45]   # L Knee
        base_offsets[5] = [0.10, 0.0, -0.45]    # R Knee
        base_offsets[6] = [0.0, 0.0, 0.30]      # Spine 2
        base_offsets[7] = [-0.10, 0.0, -0.85]   # L Ankle
        base_offsets[8] = [0.10, 0.0, -0.85]    # R Ankle
        base_offsets[9] = [0.0, 0.0, 0.45]      # Spine 3
        base_offsets[10] = [-0.10, 0.12, -0.90] # L Foot
        base_offsets[11] = [0.10, 0.12, -0.90]  # R Foot
        base_offsets[12] = [0.0, 0.0, 0.60]     # Neck
        base_offsets[13] = [-0.08, 0.0, 0.55]   # L Collar
        base_offsets[14] = [0.08, 0.0, 0.55]    # R Collar
        base_offsets[15] = [0.0, 0.0, 0.75]     # Head
        base_offsets[16] = [-0.20, 0.0, 0.52]   # L Shoulder
        base_offsets[17] = [0.20, 0.0, 0.52]    # R Shoulder
        base_offsets[18] = [-0.22, 0.0, 0.25]   # L Elbow
        base_offsets[19] = [0.22, 0.0, 0.25]    # R Elbow
        base_offsets[20] = [-0.22, 0.0, 0.00]   # L Wrist
        base_offsets[21] = [0.22, 0.0, 0.00]    # R Wrist

        # Apply subject scaling if given
        if subject_profile is not None:
            s_legs = subject_profile.get("scale_legs", 1.0)
            s_arms = subject_profile.get("scale_arms", 1.0)
            s_torso = subject_profile.get("scale_torso", 1.0)
            # Legs
            for j in [1, 2, 4, 5, 7, 8, 10, 11]: base_offsets[j, 2] *= s_legs
            # Arms
            for j in [16, 17, 18, 19, 20, 21]:
                base_offsets[j, 0] *= s_arms
                base_offsets[j, 2] = base_offsets[9, 2] + (base_offsets[j, 2] - base_offsets[9, 2]) * s_arms
            # Torso / Spine
            for j in [3, 6, 9, 12, 13, 14, 15]: base_offsets[j, 2] *= s_torso

        # Action Kinematics
        x0 = float(rng.uniform(-0.5, 0.5))
        y0 = float(rng.uniform(2.0, 3.5))
        z0 = 0.95

        c_x = np.zeros(T, dtype=np.float64)
        c_y = np.zeros(T, dtype=np.float64)
        c_z = np.zeros(T, dtype=np.float64)
        joints = np.zeros((T, 22, 3), dtype=np.float64)

        if action_type == "Normal walking":
            v_x = float(rng.uniform(-0.3, 0.3))
            v_y = float(rng.uniform(0.8, 1.2)) # ~1 m/s
            c_x = x0 + v_x * times
            c_y = y0 + v_y * times
            c_z = z0 + np.sin(times * 6.0) * 0.03
            gait_freq = 4.0
            for t in range(T):
                phase = times[t] * gait_freq
                jt = base_offsets.copy()
                jt[18, 1] += np.sin(phase) * 0.18
                jt[19, 1] -= np.sin(phase) * 0.18
                jt[4, 1] += np.cos(phase) * 0.15
                jt[5, 1] -= np.cos(phase) * 0.15
                joints[t] = np.array([c_x[t], c_y[t], c_z[t]]) + jt

        elif action_type == "Standing":
            c_x = x0 + np.sin(times * 0.5) * 0.02
            c_y = y0 + np.cos(times * 0.4) * 0.02
            c_z = z0 + np.sin(times * 0.8) * 0.005
            for t in range(T):
                joints[t] = np.array([c_x[t], c_y[t], c_z[t]]) + base_offsets

        elif action_type == "Turning":
            # Circular turning motion with angular velocity
            radius = 1.2
            omega = 1.2 # rad/s
            c_x = x0 + radius * np.sin(omega * times)
            c_y = y0 + radius * (1.0 - np.cos(omega * times))
            c_z = z0 + np.sin(times * 5.0) * 0.02
            for t in range(T):
                heading = omega * times[t]
                rot = np.array([
                    [np.cos(heading), -np.sin(heading), 0],
                    [np.sin(heading), np.cos(heading), 0],
                    [0, 0, 1]
                ])
                joints[t] = np.array([c_x[t], c_y[t], c_z[t]]) + (rot @ base_offsets.T).T

        elif action_type == "Sitting":
            # Deceleration and descent to z=0.50
            t_half = times[-1] * 0.6
            progress = np.clip(times / t_half, 0.0, 1.0)
            c_x = x0 + progress * 0.1
            c_y = y0 + progress * 0.2
            c_z = z0 - progress * 0.45 # drop 45 cm
            for t in range(T):
                jt = base_offsets.copy()
                # Flex hips & knees
                jt[4:6, 1] += progress[t] * 0.25 # knees forward
                jt[4:6, 2] += progress[t] * 0.15
                jt[7:9, 1] += progress[t] * 0.05
                joints[t] = np.array([c_x[t], c_y[t], c_z[t]]) + jt

        elif action_type == "Rising":
            # Rising from sitting
            progress = np.clip(times / (times[-1] * 0.6), 0.0, 1.0)
            c_x = x0 + progress * 0.1
            c_y = y0 + progress * 0.2
            c_z = (z0 - 0.45) + progress * 0.45
            for t in range(T):
                jt = base_offsets.copy()
                inv_p = 1.0 - progress[t]
                jt[4:6, 1] += inv_p * 0.25
                jt[4:6, 2] += inv_p * 0.15
                joints[t] = np.array([c_x[t], c_y[t], c_z[t]]) + jt

        elif action_type == "Running":
            # High speed ~3.0 m/s, large limb excursions
            v_y = float(rng.uniform(2.5, 3.2))
            c_x = x0 + float(rng.uniform(-0.1, 0.1)) * times
            c_y = y0 + v_y * times
            c_z = z0 + np.abs(np.sin(times * 9.0)) * 0.08
            gait_freq = 9.0
            for t in range(T):
                phase = times[t] * gait_freq
                jt = base_offsets.copy()
                jt[18, 1] += np.sin(phase) * 0.35
                jt[19, 1] -= np.sin(phase) * 0.35
                jt[4, 1] += np.cos(phase) * 0.30
                jt[5, 1] -= np.cos(phase) * 0.30
                joints[t] = np.array([c_x[t], c_y[t], c_z[t]]) + jt

        elif action_type == "Jumping":
            # Parabolic ballistic flight: peak vertical acc ~ 10 m/s^2, z_peak +0.5m
            c_x = x0 + 0.5 * times
            c_y = y0 + 0.8 * times
            # Jump launch at t=0.25*T, land at t=0.75*T
            t_mid = times[T // 2]
            t_rel = times - t_mid
            c_z = z0 + np.maximum(0.0, 0.45 - 3.5 * (t_rel ** 2))
            for t in range(T):
                jt = base_offsets.copy()
                if c_z[t] > z0 + 0.05: # airborne
                    jt[4:6, 2] += 0.15 # tuck legs
                    jt[7:9, 2] += 0.20
                    jt[18:20, 2] += 0.20 # arms raised
                joints[t] = np.array([c_x[t], c_y[t], c_z[t]]) + jt

        elif action_type == "Fast directional changes":
            # Rapid 180 deg direction reversal with high lateral jerk
            c_x = x0 + 1.2 * np.sin(times * 4.0)
            c_y = y0 + 1.0 * np.cos(times * 2.0)
            c_z = z0 + np.sin(times * 6.0) * 0.03
            for t in range(T):
                joints[t] = np.array([c_x[t], c_y[t], c_z[t]]) + base_offsets

        else: # "Other high-dynamic motions"
            # Agile athletic motion with high angular acceleration
            c_x = x0 + 1.5 * np.sin(times * 3.5)
            c_y = y0 + 1.8 * times
            c_z = z0 + np.abs(np.sin(times * 7.0)) * 0.15
            for t in range(T):
                phase = times[t] * 7.0
                jt = base_offsets.copy()
                jt[18:22] += rng.normal(0.0, 0.05, size=(4, 3))
                jt[4:8] += rng.normal(0.0, 0.05, size=(4, 3))
                joints[t] = np.array([c_x[t], c_y[t], c_z[t]]) + jt

        # Center trajectories & velocities
        centers = np.stack([c_x, c_y, c_z], axis=-1)
        velocities = np.zeros_like(joints)
        velocities[1:] = (joints[1:] - joints[:-1]) / self.dt
        velocities[0] = velocities[1]

        # Root velocities
        root_vel = (centers[1:] - centers[:-1]) / self.dt
        root_vel = np.vstack([root_vel[0:1], root_vel])

        # True radial velocities
        v_r_true = np.array([
            RadarMeasurementModel.compute_radial_velocity(centers[t], root_vel[t])
            for t in range(T)
        ])

        return {
            "gt_joints": joints,         # [T, 22, 3]
            "gt_centers": centers,       # [T, 3]
            "gt_velocities": velocities, # [T, 22, 3]
            "gt_v_r": v_r_true,          # [T]
            "action_type": action_type,
            "T": T,
        }

    def generate_sparsity_mask(self, T: int, density_pct: float) -> np.ndarray:
        """Generates observation mask with specified observation density (100% down to 1%)."""
        if density_pct >= 100.0:
            return np.ones(T, dtype=np.int32)
        num_obs = max(1, int(round(T * density_pct / 100.0)))
        mask = np.zeros(T, dtype=np.int32)
        indices = np.linspace(0, T - 1, num_obs, dtype=int)
        mask[indices] = 1
        return mask

    def generate_dropout_pattern_mask(self, T: int, pattern: str, pct: float) -> np.ndarray:
        """Generates masks for comparing dropout topologies."""
        num_drop = int(round(T * pct / 100.0))
        num_drop = max(1, min(T - 1, num_drop))
        mask = np.ones(T, dtype=np.int32)

        if pattern == "Random dropout":
            drop_idx = self.rng.choice(T, size=num_drop, replace=False)
            mask[drop_idx] = 0
            if np.all(mask == 0): mask[0] = 1

        elif pattern == "Contiguous dropout":
            start = max(1, (T - num_drop) // 2)
            mask[start:min(T, start + num_drop)] = 0

        elif pattern == "Multiple contiguous gaps":
            gap_len = max(1, num_drop // 2)
            mask[1:1 + gap_len] = 0
            mid = T // 2
            mask[mid:mid + gap_len] = 0

        elif pattern == "Leading gap":
            mask[0:num_drop] = 0
            mask[-1] = 1

        elif pattern == "Trailing gap":
            mask[T - num_drop:T] = 0
            mask[0] = 1

        return mask
