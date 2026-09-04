# V9.0 Architecture Freeze & Dimensional Audit — Comprehensive Research Document

## 1. Objective
This document records the rigorous pre-training freeze audit of the proposed **PhotonShield V9-Mamba Residual Physics Corrector**.
The objective of Phase V9.0 is to formally audit the architecture for mathematical correctness, dimensional consistency, physical validity, embedded deployment budget compliance, and compatibility with the frozen V8.1 mathematical estimator and M4Human pose representation.

## 2. V9 Architecture Overview
The V9 architecture is a **physics-guided neural residual learner**. Rather than predicting human 3D pose end-to-end from raw radar point clouds, the network predicts strictly the analytical discrepancy of the frozen V8.1 mathematical state estimator:
$$\hat{y}_t = y_{\text{math}, t} + \alpha_t \cdot \hat{r}_t$$
where:
- $y_{\text{math}, t} \in \mathbb{R}^{66}$ is the V8.1 analytical pose estimate (22 joints $\times$ 3D coordinates).
- $\hat{r}_t \in \mathbb{R}^{66}$ is the bounded neural residual correction.
- $\alpha_t \in [0, 1]$ is the dynamic confidence gating factor.
- The combined pose $\hat{y}_t$ is unconditionally projected through the deterministic forward kinematic tree and Rodrigues angle cone projection layer, guaranteeing $0.0\%$ bone length violations.

## 3. Input Interface (136-D)
Locked in `v9_input_dimension_audit.json`:
1. Analytical joint positions $p_t$ (66-D, meters)
2. Analytical joint velocities $v_t$ (66-D, meters/second)
3. Positional covariance trace $\text{Tr}(P_{\text{pos}, t})$ (1-D, meters$^2$)
4. Dynamic observation confidence $c_t$ (1-D, dimensionless $[0, 1]$)
5. Radar Doppler radial velocity innovation $e_{r, t}$ (1-D, meters/second)
6. Binary observation mask $m_t$ (1-D, binary $\{0, 1\}$)
Formula: $66 + 66 + 1 + 1 + 1 + 1 = 136$. **PASS**.

## 4. State Decomposition ($s_{\text{math}} = 132$)
Documented in `v9_smath_decomposition.json`:
- 22 joints $\times$ 6 kinematic variables per joint ($3$ Cartesian positions $+ 3$ Cartesian velocities) $= 132$.
- No hidden, latent, or unobserved states.

## 5. Joint Mapping (22 SMPL-X Joints)
Documented in `v9_joint_mapping.json`:
- Exact correspondence with M4Human 22 body joints: Pelvis (0), L/R Hip (1, 2), Spine 1/2/3 (3, 6, 9), L/R Knee (4, 5), L/R Ankle (7, 8), L/R Foot (10, 11), Neck (12), Head (15), L/R Collar (13, 14), L/R Shoulder (16, 17), L/R Elbow (18, 19), L/R Wrist (20, 21).
- Coordinate order: $(x, y, z)$ per joint $\implies 22 \times 3 = 66$. No duplicate or omitted joints. **PASS**.

## 6. Coordinate Convention
Documented in `v9_coordinate_audit.md`:
- Right-Handed Cartesian frame: $+X$ right, $+Y$ forward, $+Z$ up.
- Handedness: $\mathbf{\hat{x}} \times \mathbf{\hat{y}} = \mathbf{\hat{z}}$.
- Units: Meters (m).
- VoD automotive translation offset resolved by anchoring directly in M4Human metric frame. **PASS**.

## 7. Residual Target
Documented in `v9_residual_target_audit.json`:
- $r_t = y_t - y_{\text{math}, t} \in \mathbb{R}^{66}$.
- Zero-centered across training set ($|\text{mean}| < 0.005\text{ m}$).
- Total Euclidean residual norm: Mean $= 382.9\text{ mm}$, P95 $= 751.5\text{ mm}$, Max $= 4076.4\text{ mm}$. **PASS**.

## 8. Covariance Formulation & Dimensional Audit
Documented in `v9_covariance_audit.md`:
- **Audit Discovery**: Full state covariance trace $\text{Tr}(P_t)$ in V8.1 summed positional variance ($\text{m}^2$) and velocity variance ($\text{m}^2/\text{s}^2$).
- **Modification**: Redefined uncertainty feature to strictly positional covariance trace:
  $$\text{Tr}(P_{\text{pos}}) = \sum_{j=0}^{21} \sum_{k=0}^2 P_{6j+k, 6j+k} \quad [\text{m}^2]$$
  such that $\sqrt{\text{Tr}(P_{\text{pos}})}$ has strict units of meters ($\text{m}$). **PASS WITH MODIFICATION**.

## 9. Safety Bound Formulation
Documented in `v9_residual_bound_audit.json`:
- Selected Candidate D (Hybrid Covariance + Hard Cap):
  $$r_{\max, t} = \min\left( 0.5 \sqrt{\text{Tr}(P_{\text{pos}, t})} + 0.08, \, 0.25 \right) \quad [\text{m}]$$
- Provides $98.6\%$ coordinate coverage under normal motion while capping maximum correction at $0.25\text{ m}$. **PASS WITH MODIFICATION**.

## 10. Confidence Gating
Documented in `v9_confidence_gate_audit.json`:
$$\alpha_t = \sigma\left(10 \cdot (c_t - 0.20)\right) \cdot \left(1 - \text{clamp}\left(\frac{N_{\text{gap}}}{32}, 0, 1\right)\right)$$
- Strictly bounded within $[0, 1]$.
- Derives $N_{\text{gap}}$ causally from observation mask $m_t$. **PASS**.

## 11. Temporal Architecture & Causality
Documented in `v9_tensor_shapes.md`:
- Temporal window: $T = 16$ frames ($533.3\text{ ms}$ at $30\text{ Hz}$).
- **Causal Standard Frozen**: Causal convolution padding and forward-only S6 scan are frozen for real-time edge execution ($O(1)$ state updates at runtime). **PASS WITH MODIFICATION**.

## 12. BINN Constraint Formulation
Documented in `v9_binn_audit.md`:
- Hybrid design: Differentiable soft penalty losses during training; deterministic POCS projection at inference.
- Trainable BINN parameters: **0**. **PASS**.

## 13. Gradient Flow
- End-to-end differentiable backpropagation verified without NaN/Inf gradients. **PASS**.

## 14. Failure Containment
Documented in `v9_failure_containment.md`:
- Hard bypass: $\hat{r}_t = 0$ on NaN/Inf or extreme confidence collapse ($c_t < 0.02$).
- Guaranteed reversion to stable V8.1 analytical estimator. **PASS**.

## 15. Parameter Accounting & Budget Compliance
Documented in `v9_parameter_audit.json`:
- Target Budget: $\le 25,000$ parameters (MICRO embedded target).
- **Modification**: Set $d_{\text{inner}} = 47$ (expand $= 47/64$) and removed biases on linear stem and residual head.
- **Final Parameter Count**: **24777 parameters** (96.79 KB FP32).
- Margin under budget: **223 parameters**. **PASS WITH MODIFICATION**.

## 16. Compute Accounting
Documented in `v9_deployment_audit.json`:
- Multiply-Accumulates: **~23,755 MACs/frame**.
- Floating-Point Operations: **~50,774 FLOPs/frame** (~$0.051\text{ MFLOPs}$).
- Activation memory: **~2.8 KB/frame**.
- Peak RAM footprint: **~137.29 KB**. **PASS**.

## 17. Deployment Feasibility
- Fully compatible with microcontroller SRAM/Flash constraints (Arduino UNO Q / Cortex-M). **PASS**.

## 18. Audit Findings Matrix
All 21 audit items evaluated and confirmed:
- 16 PASS
- 5 PASS WITH MODIFICATION (Covariance units, Safety bound, Temporal causality, Parameter budget, Architecture freeze)
- 0 FAIL.

## 19. Architecture Modifications Summary
1. **Covariance Units**: Replaced mixed $\text{Tr}(P_t)$ with positional $\text{Tr}(P_{\text{pos}})$ ($m^2$).
2. **Safety Bound**: Upgraded to Hybrid Covariance Cap: $r_{\max} = \min(0.5\sqrt{\text{Tr}(P_{\text{pos}})}+0.08, 0.25)$.
3. **Parameter Reduction**: Set $d_{\text{inner}} = 47$, removed stem/head biases $\implies 24777$ parameters ($\le 25,000$).
4. **Causal Standardization**: Causal convolution and S6 recurrence frozen for zero future-frame leakage.

## 20. Final Frozen Architecture Specification
### PhotonShield V9-Mamba Residual Physics Corrector
- **Input Dimension**: 136-D $[s_{\text{math}}(132), \text{Tr}(P_{\text{pos}})(1), c_t(1), e_r(1), m_t(1)]$
- **Temporal Context**: $T = 16$ frames ($533.3\text{ ms}$, Causal)
- **Stem**: `nn.Linear(136, 64, bias=False)` $\to$ `nn.LayerNorm(64)` $\to$ `nn.SiLU()`
- **Core Temporal Engine**: Causal Selective SSM / Mamba (`d_model=64`, `d_state=16`, `d_inner=47`, `d_conv=3`)
- **Residual Output Head**: `nn.Linear(64, 66, bias=False)`
- **Uncertainty Output Head**: `nn.Linear(64, 1, bias=True)` $\to$ `nn.Softplus()`
- **Safety Mechanism**: Dynamic Saturation $r_{\max} = \min(0.5\sqrt{\text{Tr}(P_{\text{pos}})}+0.08, 0.25\text{ m})$
- **Gating Mechanism**: $\alpha_t = \sigma(10(c_t - 0.20)) \cdot (1 - \text{clamp}(N_{\text{gap}}/32, 0, 1))$
- **Physical BINN**: Deterministic Forward Kinematic Tree + Rodrigues Cone Projection
- **Total Parameters**: **24777 parameters**
- **Memory**: **96.79 KB FP32**
- **Compute**: **~50,774 FLOPs/frame** (~$0.051\text{ MFLOPs}$)
- **Deployment Target**: **MICRO** (Arduino UNO Q / Cortex-M / Edge MCU)
- **Architecture Status**: **FROZEN**
- **Neural Training**: **NOT STARTED**
