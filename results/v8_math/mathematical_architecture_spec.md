# PhotonShield AI — Mathematical Architecture Specification (V8-MATH)

This specification defines the complete state estimation equations for radar-based human motion reconstruction,
categorizing each component for future neural integration.

---

## 1. Mathematical Architecture Equations

### Equation 1: State Variables
$$s_t = \begin{bmatrix} p_t \\ v_t \end{bmatrix} \in \mathbb{R}^{132}, \quad p_t \in \mathbb{R}^{66}, \, v_t \in \mathbb{R}^{66}$$
- **Classification:** `HARD PHYSICS`
- **Definition:** 3D positions and linear velocities across all $J=22$ anatomical joints.

### Equation 2: State Transition Equations
$$s_{t+1} = F s_t + w_t, \quad F = \begin{bmatrix} I_{3J} & \Delta t \cdot I_{3J} \\ 0 & I_{3J} \end{bmatrix}$$
- **Classification:** `HARD PHYSICS`
- **Definition:** Discrete-time kinematic integration under constant velocity motion ($w_t \sim \mathcal{N}(0, Q)$).

### Equation 3: Observation Equations (Radar & LiDAR)
$$y_t = H s_t + v_t, \quad H = [I_{3J}, \, 0_{3J}], \quad v_t \sim \mathcal{N}(0, R)$$
- **Classification:** `SENSOR MODEL`
- **Definition:** Direct geometric spatial measurement mapping state position to observation space.

### Equation 4: Missing-Observation Gating Equations
$$m_t \in \{0, 1\}, \quad \hat{x}_{t|t} = \begin{cases} \hat{x}_{t|t-1} + K_t (y_t - H \hat{x}_{t|t-1}) & \text{if } m_t = 1 \\ \hat{x}_{t|t-1} & \text{if } m_t = 0 \end{cases}$$
- **Classification:** `ESTIMATION THEORY`
- **Definition:** Complete elimination of artificial zero-token impulse shocks by bypassing measurement updates when $m_t=0$.

### Equation 5: Covariance & Uncertainty Equations
$$P_{t|t-1} = F P_{t-1|t-1} F^T + Q, \quad P_{t|t} = \begin{cases} (I - K_t H) P_{t|t-1} (I - K_t H)^T + K_t R K_t^T & \text{if } m_t = 1 \\ P_{t|t-1} & \text{if } m_t = 0 \end{cases}$$
- **Classification:** `ESTIMATION THEORY`
- **Definition:** Rigorous state uncertainty propagation guaranteeing monotonic confidence growth over missing observation horizons.

### Equation 6: Human Skeletal Constraints
$$L_{\min, ij} \le \|p_i - p_j\|_2 \le L_{\max, ij}, \quad \theta_{\min, ijk} \le \arccos\left(\text{clamp}\left(\frac{u \cdot v}{\|u\| \|v\|}, -1, 1\right)\right) \le \theta_{\max, ijk}$$
- **Classification:** `HUMAN CONSTRAINT`
- **Definition:** Invariant anatomical bounds bounding limb lengths and joint articulation limits.

### Equation 7: Radar Radial Velocity Constraint
$$v_{r, t} = \hat{r}_t^T v_t = \frac{p_t^T v_t}{\|p_t\|_2}, \quad e_r = v_{r, \text{meas}} - \hat{r}_t^T v_t$$
- **Classification:** `SENSOR MODEL`
- **Definition:** Non-linear projection of 3D velocity onto line-of-sight Doppler measurement.

### Equation 8: LiDAR Fusion Equations
$$K_{L, t} = P_{t|t-1} H_L^T (H_L P_{t|t-1} H_L^T + R_L)^{-1}, \quad \hat{x}_{t|t} = \hat{x}_{t|t-1} + K_{L, t} (y_{L, t} - H_L \hat{x}_{t|t-1})$$
- **Classification:** `SENSOR MODEL`
- **Definition:** Optimal closed-form linear fusion of geometric LiDAR coordinates.

### Equation 9: Computational Complexity
$$\mathcal{O}(M^3) \text{ for Kalman update where } M=3J=66. \quad \text{Latency } < 0.20 \text{ ms per frame on CPU.}$$
- **Classification:** `ESTIMATION THEORY`

### Equation 10: Components Suitable for Neural Parameterization
1. **Adaptive Process Noise Covariance $Q(t)$**: Dynamic prediction of maneuvering agility (walking vs sprint).
2. **Measurement Noise Covariance $R(t)$**: Uncertainty estimation conditioned on radar SNR and point sparsity.
3. **Observation-Gated Recurrent SSM**: Neural state-space layer implementing $h_t = h_{t-1}$ when $m_t=0$.
- **Classification:** `POTENTIAL NEURAL COMPONENT`
