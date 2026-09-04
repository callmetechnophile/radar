# V9 Rigorous Evaluation Protocol & Statistical Benchmark Criteria

## 1. Metrics Suite
1. **Clean MPJPE (mm):** Accuracy under complete observations.
2. **Long-Gap MPJPE (16f, 32f mm):** Performance during missing horizons.
3. **Procrustes Aligned MPJPE (PA-MPJPE mm):** Rigid alignment invariant error.
4. **Bone Length Violation Rate (%):** Strictly required to be 0.0%.
5. **Joint Angle Violation Rate (%):** Target < 10.0%.
6. **Velocity & Acceleration MAE (m/s, m/s²):** Physical kinematic smoothness.
7. **Radar Doppler Consistency (m/s):** Mean radial innovation.
8. **Uncertainty Calibration ($r$):** Covariance-error correlation.
9. **Latency & Compute:** Frame runtime on CPU (ms), MACs, FLOPs.

## 2. Multi-Seed Statistical Validation
Every neural experiment must be evaluated across 3 random seeds (42, 123, 456).
Report: $\text{Mean} \pm \text{Std}$, 95% confidence intervals, and two-sample t-test $p$-values.
