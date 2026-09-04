# Dual-Source Uncertainty Architecture Specification

## 1. Dual Uncertainty Hierarchy
Rather than replacing the analytical Kalman covariance, V9 introduces a **dual uncertainty model**:
1. **Aleatoric Uncertainty $\Sigma_{\text{math}}$ (Analytical):**
   Supplied directly from the linear EKF covariance matrix $\text{Tr}(P_t)$. Quantifies sensor measurement sparsity and temporal propagation drift.
2. **Epistemic Residual Dispersion $\hat{\sigma}_{\text{neural}, t}$ (Learned Head):**
   Predicted by a lightweight auxiliary head: `Linear(d_model, 1) -> Softplus`. Quantifies the neural network's confidence in its own non-linear correction.

## 2. Combined Predictive Uncertainty
$$\sigma_{\text{total}, t}^2 = \text{Tr}(P_t) + \hat{\sigma}_{\text{neural}, t}^2$$
Guarantees that uncertainty is strictly positive, grounded by analytical physics, and increases monotonically during unobserved horizons.
