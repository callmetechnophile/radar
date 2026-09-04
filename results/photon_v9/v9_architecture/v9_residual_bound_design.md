# Bounded Residual Correction Safety Mechanism

## 1. Motivation
Unconstrained neural networks can extrapolate catastrophically when encountering out-of-distribution radar noise or extreme gap horizons.

## 2. Dynamic Covariance-Dependent Saturation
The raw neural residual $\tilde{r}_t \in \mathbb{R}^{66}$ is passed through an uncertainty-scaled hyperbolic tangent saturation function:
$$\hat{r}_t = r_{\max, t} \cdot \tanh\left( \frac{\tilde{r}_t}{r_{\max, t}} \right)$$
where the maximum permissible correction bound $r_{\max, t}$ is dynamically coupled to the analytical state covariance:
$$r_{\max, t} = \kappa \cdot \sqrt{\text{Tr}(P_t)} + r_{\text{base}}$$
- $r_{\text{base}} = 0.08\text{ m}$ ($8\text{ cm}$ maximum correction under clean tracking)
- $\kappa = 0.5$
- $\tanh$ guarantees that $\|\hat{r}_t\| \le r_{\max, t}$ strictly for all inputs.
