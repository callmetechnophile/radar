# View-of-Delft (VoD) to PhotonShield V5.4 Temporal Mamba Interface Precheck

## 1. Frozen V5.4 Input Interface
- **Input Tensor**: `[B, T, 64]` of type `torch.float32`
- **Mask Tensor**: `[B, T, 1]` of type `torch.float32` (1=observed, 0=missing)
- **Temporal Sequences**: Sliding temporal windows of $T \in \{4, 8, 16\}$
- **Internal Backbone**: 2-layer MiniMambaBlock ($64 \to 64$, $d_{\text{state}}=16, d_{\text{conv}}=4$)
- **Auxiliary Physics Target**: 5-DoF Planar Kinematics $[\Delta x, \Delta y, v_x, v_y, \omega]$

## 2. VoD Native Representation vs. V5.4 Interface

| Feature Dimension | VoD Native Radar | V5.4 Target Interface | Required Adapter Mapping |
| :--- | :--- | :--- | :--- |
| **Representation** | 3D Point Cloud `(N, 7)` | Continuous Vector `(64,)` | `VoDRadarFeatureAdapter` (Polar BEV spatial/Doppler pooling) |
| **Coordinates** | Cartesian $[x, y, z]$ (meters) | Compact 64-D Latent | Azimuth-range grid voxelization ($8 \times 8 = 64$) |
| **Doppler Velocity** | Direct $[v_r, v_{r,\text{comp}}]$ | Feature channels $30..60$ | Radial & compensated Doppler summary statistics |
| **Reflection Power** | Direct $\text{RCS}$ (dBsm) | Feature channels $0..30$ | RCS energy density histogram |
| **Sampling Rate** | $13.0\text{ Hz}$ ($\Delta t = 76.9\text{ ms}$) | $4.0-30.0\text{ Hz}$ | Direct temporal sequence sliding window adapter |

## 3. Compatibility Verdict
> **VERIFIED COMPATIBLE**: VoD native 3D radar point clouds contain exact physical observables (spatial coordinates $[x,y,z]$, reflection $\text{RCS}$, and Doppler velocity $v_r$) that map cleanly and deterministically into the 64-D temporal feature interface consumed by frozen V5.4 Temporal Mamba.
