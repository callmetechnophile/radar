# PhotonShield AI — Phase V5.4 Oxford Coordinate Frame & Physical Modality Audit

- **Audit Subject**: Oxford Radar RobotCar Physical Coordinates, Kinematics & Supervision Observables
- **Dataset**: `oxford_radar_robotcar_dataset_sample_medium/2019-01-10-14-36-48-radar-oxford-10k-partial`
- **Audit Date**: 2026-08-15
- **Status**: **VERIFIED & CALIBRATED**

---

## 1. Physical Modality & Ground-Truth Kinematics Inventory

| Modality Source | Path | Raw Observables | Physical Units | Synchronization Error |
| :--- | :--- | :--- | :---: | :---: |
| **Radar Ground Truth Odometry** | `gt/radar_odometry.csv` | Relative 6-DoF rigid-body motion $(\Delta x, \Delta y, \Delta z, \Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw})$ | $\text{m}, \text{rad}$ | **`0.00 ms`** (Exact radar timestamp matching) |
| **GPS / Inertial Navigation (INS)** | `gps/ins.csv` | Global Northing/Easting, $(v_{\text{north}}, v_{\text{east}}, v_{\text{down}})$, $(\text{roll}, \text{pitch}, \text{yaw})$ | $\text{m}, \text{m/s}, \text{rad}$ | **`< 2.00 ms`** (Linear timestamp interpolation) |
| **Stereo Visual Odometry** | `vo/vo.csv` | Camera frame relative poses | $\text{m}, \text{rad}$ | Verified auxiliary |

---

## 2. Coordinate Frames & Transformations

### A. Oxford Vehicle Reference Frame ($\mathcal{F}_{\text{vehicle}}$)
- **Origin**: Center of rear axle projected to ground plane.
- **Axes**:
  - $+X_{\text{vehicle}}$: Forward along vehicle longitudinal driving axis.
  - $+Y_{\text{vehicle}}$: Left along lateral axis.
  - $+Z_{\text{vehicle}}$: Upwards orthogonal to ground plane.

### B. Navtech CTS350-X Radar Frame ($\mathcal{F}_{\text{radar}}$)
- **Mounting**: Roof rack center, facing forward.
- **Polar Coordinate Mapping**:
  - Azimuth $\theta \in [0, 2\pi)$ sampled across 400 discrete beams ($0.90^\circ/\text{step}$).
  - Range $r \in [0, 162.78\text{ m}]$ sampled across 3768 bins ($0.0432\text{ m/bin}$).
- **Kinematic Equivalence**: Planar rigid-body translation $\Delta p = [\Delta x, \Delta y]^\top$ and planar yaw rotation $\Delta \theta$ align directly with vehicle longitudinal/lateral planar motion.

---

## 3. Kinematic Equations for V5.4 Supervision

Given radar sequence timestamps $t_0, t_1, \dots, t_{T-1}$ with sampling interval $\Delta t_t = t_{t+1} - t_t$ (mean $\Delta t \approx 250.08\text{ ms}$):

### 1. Relative Inter-Frame Motion Target:
$$\Delta p_t = [x_{t+1} - x_t, y_{t+1} - y_t]^\top \in \mathbb{R}^2$$

### 2. Velocity Target:
$$v_t = \frac{\Delta p_t}{\Delta t_t} \in \mathbb{R}^2 \quad (\text{m/s})$$

### 3. Acceleration Target:
$$a_t = \frac{v_{t+1} - v_t}{\Delta t_t} \in \mathbb{R}^2 \quad (\text{m/s}^2)$$

### 4. Yaw Rate Target:
$$\omega_t = \frac{\Delta \text{yaw}_t}{\Delta t_t} \quad (\text{rad/s})$$

---

## 4. Physics Supervision Target ($K = 5$ observables)

For each radar timestep $t$, the ground-truth kinematic supervision vector is defined as:
$$y_t^{\text{phys}} = \big[\Delta x_t, \Delta y_t, v_{x,t}, v_{y,t}, \omega_t\big]^\top \in \mathbb{R}^5$$

- **Inference Guarantee**: The model receives **RADAR ONLY** at inference. Odometry and kinematic ground truth are utilized **strictly as auxiliary loss supervision** during training.
