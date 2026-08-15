# PhotonShield AI -- Phase V5.0 Oxford Radar RobotCar Temporal Data Audit

- **Audited Dataset**: Oxford Radar RobotCar Dataset (`2019-01-10-14-36-48-radar-oxford-10k-partial`)
- **Total Radar Scans**: **`51`** scans
- **Total Sequence Duration**: **`12.51 seconds`**
- **Effective Radar FPS**: **`4.00 Hz`** (Mean dt = `250.12 ms`, Jitter = `1.59 ms`)
- **Native Scan Representation**: **`400 azimuths` x `3768 range bins`** (Range res = `0.0432 m/bin`, Max range = `162.78 m`)
- **Ground Truth Odometry**: **`Available`** (Synchronized with radar timestamps, mean sync error = `0.00 ms`)

## 1. Dataset Modality Inventory

| Modality | Sensor Model | Directory | Format | Availability |
| :--- | :--- | :---: | :---: | :---: |
| **FMCW Radar** | Navtech CTS350-X (76-77 GHz) | `radar/` | 8-bit PNG (400x3779) | **`Present (51 scans)`** |
| **Radar Odometry** | Ground Truth Odometry | `gt/` | CSV | **`Present (1,123 poses)`** |
| **Visual Odometry** | Stereo VO Pipeline | `vo/` | CSV | **`Present (2,842 poses)`** |
| **3D LiDAR (Left/Right)** | Velodyne HDL-32E | `velodyne_left/`, `velodyne_right/` | Binary | **`Present`** |
| **2D LiDAR (Front/Rear)** | SICK LMS-151 | `lms_front/`, `lms_rear/` | Binary | **`Present`** |
| **Stereo Camera** | Point Grey Bumblebee XB3 | `stereo/` | PNG / Timestamps | **`Present`** |
| **Mono Cameras** | Point Grey Grasshopper2 | `mono_left/`, `mono_rear/`, `mono_right/` | PNG / Timestamps | **`Present`** |
| **GPS / INS** | NovAtel SPAN-CPT | `gps/` | CSV | **`Present`** |

---

## 2. Sliding Temporal Window Analysis

| Sequence Length T | Valid Sequences | Rejected Sequences | Mean Duration (s) | Mean Interval $\Delta t$ (ms) |
| :---: | :---: | :---: | :---: | :---: |
| **T = 4** | **`48`** | `0` | `0.75 s` | `250.08 ms` |
| **T = 8** | **`44`** | `0` | `1.75 s` | `250.07 ms` |
| **T = 16** | **`36`** | `0` | `3.75 s` | `250.08 ms` |

---

## 3. RaDICaL vs. Oxford Radar Representation Comparison

| Feature Dimension | RaDICaL (Indoor / Controlled) | Oxford Radar RobotCar (Automotive / Urban) |
| :--- | :--- | :--- |
| **Physical Sensor** | TI IWR1443 FMCW (77 GHz) | Navtech CTS350-X Scanning FMCW (76-77 GHz) |
| **Scan Geometry** | Fixed Patch / MIMO Range-Doppler | $360^\circ$ Continuous Mechanical Azimuth Sweep |
| **Dimensions** | $[B, T=16, D=64]$ Range-Doppler FFT | $[B, T, 400, 3768]$ Polar / $[B, T, 640, 640]$ Cartesian |
| **Frame Rate** | $\approx 30\text{ Hz}$ ($\Delta t \approx 33.3\text{ ms}$) | $\approx 3.98\text{ Hz}$ ($\Delta t \approx 251.2\text{ ms}$) |
| **Maximum Range** | $\approx 10.0\text{ m}$ (Indoor human targets) | $\approx 162.78\text{ m}$ (Long-range urban environment) |
| **Doppler Velocity** | Explicit Doppler FFT bins | Implicit via inter-frame temporal kinematics & odometry |
| **Ground Truth Motion** | Human action labels & target presence | Metric 6-DoF vehicle odometry poses ($x, y, z, \text{yaw}$) |
