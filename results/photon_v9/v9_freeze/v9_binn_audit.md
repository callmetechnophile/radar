# V9 BINN Architecture & Constraint Integration Audit

## 1. Dual-Stage Hybrid Formulation
The Boundary-Informed Neural Network (BINN) integration operates in two distinct stages:
1. **Training Stage (Soft Kinematic Losses)**:
   Differentiable penalty regularizers guide gradient descent without creating hard, non-differentiable bottlenecks in backpropagation.
2. **Inference Stage (Deterministic POCS Projection)**:
   The neural residual output unconditionally passes through frozen analytical forward kinematic tree and Rodrigues angle cone projections, mathematically guaranteeing **0.0% bone length violations**.

## 2. Loss Formulations for Training
$$\mathcal{L}_{\text{total}} = \lambda_p \mathcal{L}_{\text{pose}} + \lambda_b \mathcal{L}_{\text{bone}} + \lambda_j \mathcal{L}_{\text{joint}} + \lambda_v \mathcal{L}_{\text{vel}} + \lambda_a \mathcal{L}_{\text{acc}} + \lambda_r \mathcal{L}_{\text{radar}} + \lambda_u \mathcal{L}_{\text{unc}}$$

- **Pose Reconstruction**: $\mathcal{L}_{\text{pose}} = \text{SmoothL1}(\hat{y}_t - y_t)$
- **Bone Invariance**: $\mathcal{L}_{\text{bone}} = \sum_{(u, v) \in \mathcal{T}} \| \|\hat{y}_u - \hat{y}_v\|_2 - l_{uv}^* \|_2^2$
- **Joint Articulation Cones**: $\mathcal{L}_{\text{joint}} = \sum_{(u, v, w)} \text{ReLU}(\theta_{uvw} - \theta_{\max}) + \text{ReLU}(\theta_{\min} - \theta_{uvw})$
- **Velocity Regularization**: $\mathcal{L}_{\text{vel}} = \sum_{j} \text{ReLU}(\|\hat{v}_j\|_2 - v_{\max, j})$
- **Acceleration Bound**: $\mathcal{L}_{\text{acc}} = \sum_{j} \text{ReLU}(\|\hat{a}_j\|_2 - a_{\max, j})$
- **Radar Doppler Consistency**: $\mathcal{L}_{\text{radar}} = \| \mathbf{v}_{r, \text{pred}}(\hat{y}_{\text{root}}, \hat{v}_{\text{root}}) - v_{r, \text{meas}} \|_2^2$
- **Uncertainty NLL**: $\mathcal{L}_{\text{unc}} = \frac{1}{2} \log \sigma_{\text{total}}^2 + \frac{\|\hat{y} - y\|_2^2}{2 \sigma_{\text{total}}^2}$

## 3. Trainable Parameters in BINN
- BINN trainable parameters: **0 parameters**.
- All bone targets $l_{uv}^*$, angle bounds $[\theta_{\min}, \theta_{\max}]$, and kinematic matrices are frozen from V8.1.
