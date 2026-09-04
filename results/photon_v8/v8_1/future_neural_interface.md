# PhotonShield AI — Future Neural Architecture Interface Contract

## Purpose
The future deep-learning model MUST NOT re-invent kinematics or basic Kalman filtering.
Instead, the neural network acts as a **Residual Physics Corrector**, learning only non-linear agile accelerations that linear kinematics cannot explain.

---

## 1. Input Contract (Passed from Mathematical Estimator to Neural Head)
1. `s_math`: Analytical mathematical state estimate $[B, T, 132]$ (positions and velocities).
2. `unc_tr`: Propagated Kalman uncertainty trace $[B, T, 1]$.
3. `conf_c`: Observation confidence metric $c_t \in [0, 1]$ $[B, T, 1]$.
4. `e_radial`: Instantaneous radar radial velocity residual $[B, T, 1]$.
5. `obs_mask`: Binary observation mask $m_t \in \{0, 1\}$ $[B, T, 1]$.

## 2. Neural Architecture Scope
- **Learned Task:** Predict state residual $\Delta s_t = [\Delta p_t, \Delta v_t]$.
- **Conditioning:** Mask-gated Mamba SSM layer ensuring zero updates when $m_t = 0$.
- **Invariance Enforcement:** Final output passes through frozen `AdvancedSkeletalConstraints.project_global_least_squares`.
