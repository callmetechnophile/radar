# PhotonShield AI — Phase V5.0 Research Hypothesis

## 1. Core Hypothesis

> **"Temporal representation learning on the Oxford Radar RobotCar dataset will enable PhotonShield to accurately reconstruct missing long-range 2D radar scans and maintain physically consistent vehicle motion dynamics under severe sensor dropouts."**

## 2. Theoretical Motivation

1. **Extended Spatial Context**: Unlike indoor short-range radar (RaDICaL, $<10\text{ m}$), Oxford provides full $360^\circ$ spatial observations up to $162.8\text{ m}$, exposing long-term temporal persistence of landmarks, road boundaries, and moving vehicles.
2. **Kinematic Ground Truth**: Oxford contains synchronized 6-DoF vehicle odometry, allowing differentiable physical constraints to directly supervise inter-frame translation and rotation $(\Delta x, \Delta y, \Delta \theta)$.
3. **Realistic Sensor Occlusion**: The $4\text{ Hz}$ sampling rate introduces significant temporal displacement per frame, creating a challenging and realistic benchmark for gap-aware physics-informed latent diffusion.

## 3. Formal Evaluation Metrics for V5 Pipeline

1. **Missing-Frame Reconstruction MSE**: Mean squared error on unobserved spatial cells $\frac{1}{|\mathcal{M}|} \sum_{(x,y) \in \mathcal{M}} (I_{x,y} - \hat{I}_{x,y})^2$.
2. **Temporal Reconstruction MAE**: Spatial mean absolute error over reconstructed frames.
3. **Temporal Continuity Error**: Inter-frame spatial structural similarity SSIM across contiguous frames.
4. **Velocity Consistency**: Residual error between radar-estimated displacement $\frac{\Delta \mathbf{r}}{\Delta t}$ and vehicle odometry velocity $\mathbf{v}$.
5. **Acceleration Consistency**: Bounded penalty on unphysical angular and linear accelerations $\frac{\Delta \mathbf{v}}{\Delta t}$.
6. **Odometry Consistency**: Relative pose error (RPE) against Oxford ground-truth trajectory.
7. **Downstream Perception Transfer**: Feature transferability to upstream classification and object detection heads.
