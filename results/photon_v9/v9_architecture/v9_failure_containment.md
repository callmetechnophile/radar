# Failure Containment & Graceful Degradation Protocol

## 1. Fault Triggers
1. NaN or Inf detected in sensor inputs or internal activations.
2. Complete observation dropout spanning $> 32$ frames ($> 1.06$ seconds).
3. Observation confidence $c_t < 0.02$.
4. Radial velocity error $|e_r| > 2.0\text{ m/s}$.

## 2. Automated Containment Actions
- **Action A (Zero Residual Override):** If NaN/Inf or extreme uncertainty occurs, force $\hat{r}_t = 0$. The output defaults identically to the V8.1 analytical estimator.
- **Action B (Skeletal Projection Enforcement):** All outputs unconditionally pass through forward kinematic tree projection, ensuring that corrupted neural weights can never dismember the skeleton.
- **Action C (Velocity Clamping):** Distal joint velocities exceeding physiological limits ($> 6.5\text{ m/s}$) are clipped to the empirical training threshold.
