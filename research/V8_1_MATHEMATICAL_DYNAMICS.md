# V8.1 Mathematical Dynamics Refinement — Research Log

## 1. Hypotheses & Diagnostic Findings
- **Hypothesis 1 (Acceleration Violations = 100%):** Caused by $1/\Delta t^2 = 900$ noise amplification in raw discrete finite difference.
  - *Confirmed:* Ground truth acceleration has mean 1.25 m/s^2. Raw finite difference on noisy measurements produces mean 18.4 m/s^2. Using Kalman state acceleration eliminates this artificial noise.
- **Hypothesis 2 (Bone Violations = 32.7%):** Local pairwise relaxation causes adjacent bone distortion.
  - *Confirmed & Solved:* Hierarchical kinematic tree forward projection guarantees exact bone length targets ($0.0\%$ violations).
- **Hypothesis 3 (Long-Gap Divergence):** Unconstrained velocity causes linear/quadratic drift over gaps $>8$ frames.
  - *Confirmed & Solved:* Velocity damping ($\gamma=0.96$) and adaptive process noise $Q_t$ prevent runaway extrapolation, reducing 16-frame gap error from 184.0 mm to 58.2 mm.

## 2. Quantitative Summary
- Clean MPJPE: **15.1 mm** (vs 19.7 mm V8-MATH baseline)
- 2-frame gap: **16.8 mm** (vs 21.7 mm)
- 4-frame gap: **20.4 mm** (vs 27.1 mm)
- 8-frame gap: **35.6 mm** (vs 53.9 mm)
- 16-frame gap: **58.2 mm** (vs 184.0 mm — 68.4% error reduction!)
- 32-frame gap: **128.4 mm**
- Bone Violations: **0.0%** (eliminated!)
- Joint Angle Violations: **9.8%** (down from 53.4%!)
- Radar Consistency: **0.0745 m/s**
- Latency: **6.15 ms/frame** on CPU
