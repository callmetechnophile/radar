# Boundary-Informed Neural Network (BINN) Integration Architecture

## 1. Architectural Placement
The BINN operates as a **Hybrid Dual-Stage Constraint Engine**:
1. **Differentiable Training Regularizer:** Penalizes violation of anatomical bone lengths, joint articulation cones, velocity limits, and acceleration envelopes during backpropagation.
2. **Deterministic Inference Projection Layer:** At inference time, the raw corrected pose candidate $p_{\text{cand}} = p_{\text{math}} + \hat{r}_p$ passes through the frozen analytical `project_global_least_squares` forward kinematic solver.

## 2. Invariance Guarantee
Because the final layer is the hierarchical kinematic tree projection outward from the Pelvis root:
$$\|p_{\text{child}} - p_{\text{parent}}\| = L_{\text{target}} \quad \forall (i, j) \in \text{Bones}$$
**Bone length violations remain strictly 0.0% by construction**, regardless of neural network weights, noise, or numerical edge cases!

## 3. Joint Angle Cones
For anatomical triplets $(u, v, w)$, if articulation angle $\theta \notin [\theta_{\min}, \theta_{\max}]$, Rodrigues rotation clamps the child bone to the cone boundary without shifting the parent.
