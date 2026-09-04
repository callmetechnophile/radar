# Confidence-Aware Residual Activation & Extreme-Horizon Control

## 1. Operating Regime Gating
Phase V8.2 established three operational zones:
- **GREEN ($\le 8$ frames):** High confidence, analytical estimator reliable, residual fine-tunes pose.
- **YELLOW ($9-24$ frames):** Moderate confidence, neural residual active to compensate for velocity drift.
- **RED ($>24$ frames):** Low confidence ($c_t \approx 0$), linear extrapolation ungrounded.

## 2. Dynamic Gating Factor $\alpha_t$
$$\hat{y}_t = a_t + \alpha_t \cdot \hat{r}_t$$
$$\alpha_t = \sigma\left( \beta \cdot (c_t - c_{\text{threshold}}) \right) \cdot (1 - \text{clamp}(N_{\text{gap}} / 32, 0, 1))$$
In the RED region ($N_{\text{gap}} > 24$), $\alpha_t \to 0$, causing the model to gracefully revert to pure damped kinematics fallback ($a_t$) rather than amplifying hallucinated athletic motions.
