# Multi-Objective Physics-Informed Loss Function Formulation

## 1. Loss Formulation
$$\mathcal{L}_{\text{total}} = \lambda_p \mathcal{L}_{\text{pose}} + \lambda_v \mathcal{L}_{\text{vel}} + \lambda_a \mathcal{L}_{\text{acc}} + \lambda_b \mathcal{L}_{\text{bone}} + \lambda_j \mathcal{L}_{\text{joint}} + \lambda_r \mathcal{L}_{\text{radar}} + \lambda_u \mathcal{L}_{\text{unc}}$$

## 2. Component Definitions
1. **Pose Loss (Smooth L1):**
   $$\mathcal{L}_{\text{pose}} = \frac{1}{T \cdot J} \sum_{t, j} \text{SmoothL1}(p_{\text{cand}, t, j} - p_{\text{gt}, t, j}, \beta=0.01)$$
2. **Velocity Kinematic Continuity Loss:**
   $$\mathcal{L}_{\text{vel}} = \frac{1}{(T-1) J} \sum_{t, j} \left\| \frac{p_{t+1, j} - p_{t, j}}{\Delta t} - v_{\text{gt}, t, j} \right\|_1$$
3. **Softplus Acceleration Bound Loss:**
   $$\mathcal{L}_{\text{acc}} = \frac{1}{T \cdot J} \sum_{t, j} \text{Softplus}\left( \frac{\|a_{t, j}\| - a_{\max, j}}{\tau} \right)$$
4. **Bone Length Invariance Loss:**
   $$\mathcal{L}_{\text{bone}} = \sum_{(u, v) \in \text{Tree}} \left( \|p_{u} - p_{v}\| - L_{uv,\text{target}} \right)^2$$
5. **Joint Angle Violation Penalty:**
   $$\mathcal{L}_{\text{joint}} = \sum_{(u, v, w)} \text{ReLU}(\theta_{\min} - \theta)^2 + \text{ReLU}(\theta - \theta_{\max})^2$$
6. **Radar Radial Velocity Consistency Loss:**
   $$\mathcal{L}_{\text{radar}} = \frac{1}{T} \sum_{t} \left| v_{r,\text{meas}, t} - \frac{p_{\text{root}, t}^T v_{\text{root}, t}}{\|p_{\text{root}, t}\|} \right|$$
7. **Negative Log-Likelihood Uncertainty Calibration Loss:**
   $$\mathcal{L}_{\text{unc}} = \frac{1}{2} \sum_{t} \left( \frac{\|e_t\|^2}{\sigma_t^2} + \log \sigma_t^2 \right)$$

## 3. Weighting Strategy
- $\lambda_p = 1.0$ (Primary metric anchor)
- $\lambda_v = 0.2$
- $\lambda_a = 0.05$
- $\lambda_b = 0.5$ (High priority on skeletal integrity)
- $\lambda_j = 0.1$
- $\lambda_r = 0.1$
- $\lambda_u = 0.05$
