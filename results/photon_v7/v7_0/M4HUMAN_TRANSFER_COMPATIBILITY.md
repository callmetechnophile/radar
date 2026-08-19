# Cross-Dataset Representation Compatibility Matrix: Oxford vs. VoD vs. M4Human

| Feature / Modality Attribute | Oxford Radar RobotCar (Dataset 1) | View-of-Delft (VoD) (Dataset 2) | M4Human (Dataset 3) | Compatibility Status | Adapter Requirement |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Sensor Type** | 2D FMCW Polar Radar (Navtech CTS350-X) | 3D High-Res mmWave Radar (ZF FRGen21) | 3D High-Res mmWave Radar (TI IWR1843BOOST) | **COMPATIBLE** | Linear Feature Projection |
| **Native Feature Dim** | Polar scan (3768 x 400) -> 64-D | Point Cloud (N x 7) -> 64-D | Point Cloud (N x 5) -> 64-D | **COMPATIBLE** | M4HumanRadarAdapter (N x 5 -> 64-D) |
| **Spatial Bounds** | Range [0, 163 m], 360 BEV | Range [0, 32 m], 100 x 30 FOV | Range [0.5, 6 m], 120 x 30 FOV | **COMPATIBLE** | Normalized Metric Coordinates |
| **Coordinate System** | Sensor BEV Cartesian (meters) | Sensor 3D Cartesian (+x fwd, +y left, +z up) | Sensor 3D Cartesian (+y depth, +x lat, +z vert) | **COMPATIBLE** | Fixed Permutation / Extrinsic Rotation |
| **Temporal Frequency** | 10.0 Hz (dt = 0.100 s) | 13.0 Hz (dt = 0.077 s) | 30.0 Hz (dt = 0.033 s) | **COMPATIBLE** | Configurable dt in Physics Loss |
| **Temporal Sequence (T)** | T = 16 frames (1.60 s) | T = 16 frames (1.23 s) | T = 16 frames (0.53 s) | **COMPATIBLE** | Identical T=16, D=64 Mamba Backbone |
| **Supervision Targets** | Trajectory & Kinematic Continuity | 3D Bounding Boxes & Classes | 22 3D Body Joints & SMPL-X Mesh | **TRANSFERABLE** | Modular Head Replacement |
| **Temporal Prior Utility** | Pretrained long-horizon motion prior | Pretrained 3D spatial-temporal prior | Articulated human kinematics | **VALIDATED** | Direct weight initialization from V6.4 |
