# PhotonShield M4Human Architecture & Task Head Decomposition

## High-Level Pipeline

`
M4Human Radar Point Cloud [B, T, N, 5]
              ↓
    M4HumanRadarAdapter (5 -> 64-D)
              ↓
  Input Projection (64 -> 64-D)
              ↓
  Pretrained Temporal Mamba (D=64, 2 layers, Bidirectional SSM)  <-- INITIALIZED FROM V6.4 VOD FOUNDATION
              ↓
  Temporal Kinematic Latents [B, T, 64]
      ↙               ↓               ↘
HumanDetectionHead   HumanPoseHead   HumanKinematicHead
(3D Human AABB)    (22 3D Body Joints)  (Velocity & Physics Regularizer)
`

## Component Parameter Breakdown

| Component | Architecture | Parameter Count | Transferred from V6.4? |
| :--- | :--- | :---: | :---: |
| **M4HumanRadarAdapter** | 2-layer MLP + LayerNorm (5 -> 32 -> 64) | 2,336 | Random Init (Task Adapter) |
| **in_proj** | Linear (64 -> 64) | 4,352 | **TRANSFERRED FROM V6.4** |
| **mamba_layers** | 2-Layer Bidirectional Mamba SSM (D=64) | 64,000 | **TRANSFERRED FROM V6.4** |
| **
orm** | LayerNorm (D=64) | 128 | **TRANSFERRED FROM V6.4** |
| **HumanDetectionHead** | 2-layer MLP (64 -> 64 -> 7 [x,y,z,l,w,h,yaw]) | 4,679 | Task Head |
| **HumanPoseHead** | 2-layer MLP (64 -> 128 -> 66 [22 joints x 3]) | 16,770 | Task Head |
| **HumanKinematicHead**| 2-layer MLP (64 -> 32 -> 4 [range, vx, vy, vz]) | 2,212 | Task Head |
| **TOTAL M4HUMAN MODEL** | Shared Backbone + Multi-Task Heads | **94,477** | **72.5% Pretrained Backbone** |
