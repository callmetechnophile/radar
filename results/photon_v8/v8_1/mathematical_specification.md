# PhotonShield AI — V8.1 Mathematical Dynamics Specification

## 1. State Vector
$$s_t = [p_t, v_t]^T \in \mathbb{R}^{132} \quad \text{(with damped kinematic propagation)}$$

## 2. State Transition & Damped Jerk-Limited Dynamics
$$s_{t+1} = F s_t + w_t, \quad F = \begin{bmatrix} I_{3J} & \Delta t \cdot I_{3J} \\ 0 & \gamma \cdot I_{3J} \end{bmatrix}, \quad \gamma = 0.96$$
Where $\gamma = 0.96$ is the velocity damping factor preventing quadratic trajectory divergence during unobserved horizons.

## 3. Adaptive Process Noise Covariance $Q_t$
$$Q_t = Q_{\text{base}} \cdot \left(1 + 0.5 \|v_t\|\right) \cdot \left(1 + 0.1 N_{\text{gap}}\right) \cdot \frac{1}{0.2 + 0.8 c_t}$$

## 4. Observation Confidence Metric $c_t$
$$c_t = \begin{cases} (0.5 m_{\text{radar}} + 0.5 m_{\text{lidar}}) \cdot \exp(-|e_r| / 0.5) & \text{if observed} \\ 0.85^{N_{\text{gap}}} & \text{if missing} \end{cases}$$

## 5. Hierarchical Skeletal Tree Projection
For each bone $(p_{\text{parent}}, p_{\text{child}})$ traversing from Pelvis outward:
$$p_{\text{child}} = p_{\text{parent}} + L_{\text{target}} \cdot \frac{p_{\text{child}} - p_{\text{parent}}}{\|p_{\text{child}} - p_{\text{parent}}\|}$$
Guarantees $L_{ij} = L_{\text{target}}$ exactly ($0.0\%$ bone violation rate).

## 6. Joint Angle Cone Projection (Rodrigues Formulation)
For articulation triplet $(u, v, w)$ with $\theta \notin [\theta_{\min}, \theta_{\max}]$:
$$u_2' = u_2 \cos(\Delta\theta) + (a \times u_2) \sin(\Delta\theta) + a(a \cdot u_2)(1 - \cos(\Delta\theta))$$
$$p_w = p_v + u_2' \cdot \|p_w - p_v\|$$
