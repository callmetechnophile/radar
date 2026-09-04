# V9 Dual-Source Uncertainty Head Audit

## 1. Separation of Physical vs Epistemic Uncertainty
To prevent redundant or conflicting uncertainty estimations, V9 strictly separates uncertainty into two complementary sources:

1. **Analytical Physical Dispersion ($\sigma_{\text{phys}}^2$)**:
   - Source: V8.1 Canonical Kalman filter positional covariance trace $\text{Tr}(P_{\text{pos}, t}) / 22$.
   - Nature: Captures observation geometry, sensor dropout duration, and linear state propagation variance.
   - Units: $\text{m}^2$.

2. **Neural Epistemic Dispersion ($\hat{\sigma}_{\text{neural}}^2$)**:
   - Source: Auxiliary neural head `Linear(64, 1)` + `Softplus()`.
   - Nature: Captures neural model confidence in its non-linear residual correction for the current motion dynamic regime.
   - Units: $\text{m}^2$.

3. **Total Fused Predictive Uncertainty**:
   $$\sigma_{\text{total}, t}^2 = \frac{1}{22} \text{Tr}(P_{\text{pos}, t}) + \hat{\sigma}_{\text{neural}, t}^2 \quad [\text{m}^2]$$

## 2. Distinction from Observation Confidence $c_t$
- $c_t \in [0, 1]$: Input feature measuring instantaneous radar/LiDAR sensor signal validity.
- $\sigma_{\text{total}}^2$: Output metric measuring total spatial confidence in the reconstructed 3D pose.
