# Oxford V5.5 to View-of-Delft (VoD) Representation Compatibility Analysis

## 1. Domain & Semantic Comparison Matrix

| Property | Oxford RobotCar (V5.5 Foundation) | View-of-Delft (VoD V6.1 / V6.2) | Compatibility Analysis |
| :--- | :--- | :--- | :--- |
| **Sensor Physics** | Navtech CTS350-X (FMCW Scanning Polar Radar) | ZF 3D Full-Range Imaging Radar (192 Channels) | **Domain Shift**: 2D polar image vs 3D point cloud |
| **Input Shape** | `(400, 3768)` Polar Scan | `(N, 7)` $[x,y,z,\text{RCS},v_r,v_{r,\text{comp}},\text{time\_id}]$ | Adapter required: `RadarPointEncoder` ($7 \to 64$) |
| **Latent Feature Dimension** | $D = 64$ continuous vector | $D = 64$ continuous token embedding | **100% Mathematically Compatible** |
| **Temporal Recurrence Backbone** | 2-Layer MiniMambaBlock ($64 \to 64$) | 2-Layer MiniMambaBlock ($64 \to 64$) | **Direct Weight Transfer Valid** |
| **Temporal Frequency** | $4.0\text{ Hz}$ ($\Delta t = 250\text{ ms}$) | $13.0\text{ Hz}$ ($\Delta t = 76.92\text{ ms}$) | Sampling rate shift ($\Delta t$ scaling factor $\approx 3.25\times$) |
| **Kinematic Output Semantics** | Planar Kinematics $[\Delta x, \Delta y, v_x, v_y, \omega]$ | Planar Kinematics + Radial Doppler | **Loss & Regularizer Transfer Compatible** |
| **Downstream Target** | Temporal Sequence Inpainting / Trajectory | 3D Bounding Boxes + Classification | Downstream 3D Head required |

---

## 2. Adaptation Strategy for Downstream VoD Transfer
1. **Direct Weight Transfer**: The Mamba selective state space matrices ($A, B, C, \Delta$, conv1d kernel weights, layer norms) can be directly loaded into downstream models because both utilize the canonical $D=64$ latent manifold.
2. **Frequency Calibration**: When applying the auxiliary kinematic physics head to VoD, update the sampling interval parameter from $\Delta t = 0.25\text{ s}$ to $\Delta t = 0.07692\text{ s}$.
3. **Point Encoder Coupling**: VoD point clouds $(N \times 7)$ are mapped through the lightweight `RadarPointEncoder` to produce the sequence tokens $[B, T, 64]$ ingested by the transferred Mamba foundation.
