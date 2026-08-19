# Oxford V5.4 to View-of-Delft (VoD) V6.2 Transfer Compatibility Audit

## 1. Architectural & Domain Comparison

| Dimension | Oxford Radar RobotCar (V5.4) | View-of-Delft (VoD V6.1 / V6.2) | Compatibility Verdict |
| :--- | :--- | :--- | :--- |
| **Sensor Type** | Navtech CTS350-X (FMCW Scanning Radar) | ZF 3D Full-Range Imaging Radar (192 Channels) | **Domain Shift**: Scan polar image vs 3D point cloud |
| **Native Data Format** | Polar scan image `(400, 3768)` | Point cloud `(N, 7)` $[x,y,z,\text{RCS},v_r,v_{r,\text{comp}},\text{time\_id}]$ | Handled via `RadarPointEncoder` ($7 \to 64$) |
| **Latent Feature Dimension** | $D = 64$ continuous feature vector | $D = 64$ continuous token embedding | **100% Mathematically Compatible** |
| **Temporal Recurrence** | 2-Layer MiniMambaBlock ($64 \to 64$) | 2-Layer MiniMambaBlock ($64 \to 64$) | **Direct Weight Transfer Valid** |
| **Sequence Length $T$** | $T = 8, 16$ | $T = 8, 16$ | **Identical Temporal Structure** |
| **Sampling Frequency** | $4.0\text{ Hz}$ ($\Delta t = 250\text{ ms}$) | $13.0\text{ Hz}$ ($\Delta t = 76.9\text{ ms}$) | Temporal rate shift (requires $\Delta t$ calibration) |
| **Coordinate System** | ISO 8855 Radar Frame ($+X$ Forward, $+Y$ Left) | ISO 8855 Radar Frame ($+X$ Forward, $+Y$ Left, $+Z$ Up) | **Compatible Horizontal Kinematics** |
| **Physics Head Targets** | 5-DoF Kinematics $[\Delta x, \Delta y, v_x, v_y, \omega]$ | 5-DoF Kinematics + Radial Doppler Observables | **Physics Loss Transfer Compatible** |
| **3D Downstream Task** | 2D BEV Inpainting / Motion Trajectory | 3D Bounding Box $[x,y,z,l,w,h,\text{yaw}]$ + Class | **New 3D Object Head Required** |

---

## 2. Transfer Feasibility & Strategy
1. **Transferred Components**:
   - `MiniMambaBlock` state-space parameters (temporal recurrence matrices $A, B, C, \Delta$, conv1d kernel weights, input/output projections).
   - Auxiliary `OxfordPhysicsHead` kinematic weights ($64 \to 32 \to 5$).
2. **VoD Adaptations**:
   - `RadarPointEncoder` maps native $(N, 7)$ point clouds to the $64$-D latent interface consumed by Mamba.
   - `VoDObject3DHead` maps $64$-D temporal latent states to 3D bounding-box coordinates and multi-class logits.
3. **Hypothesis Under Test**:
   - *Transfer-C* (Oxford Mamba + Auxiliary Physics Loss) will constrain velocity and trajectory predictions on VoD, providing superior track consistency and 3D bounding-box localization compared to training from scratch with limited sequences.
