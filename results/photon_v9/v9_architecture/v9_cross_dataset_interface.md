# Unified Cross-Dataset Architecture Interface (Oxford / VoD / M4Human)

## 1. Decoupled Sensory Ingestion
The neural residual core strictly consumes the 136-D analytical interface. It never accesses raw sensor representations directly.
This means cross-dataset differences are completely absorbed by the front-end EKF:
- **Oxford Radar RobotCar:** 2D PPI scans -> Range-Doppler centroid extraction -> EKF position update.
- **View-of-Delft (VoD):** 3D FMCW point clouds (x, y, z, RCS, Doppler) -> Point-to-centroid Kalman update.
- **M4Human:** Multi-radar synchronized cluster -> Multi-modal EKF.

## 2. Coordinate System Standardization
All coordinates are normalized to the right-handed Cartesian frame:
- $X$: Lateral (Right)
- $Y$: Longitudinal Range (Forward)
- $Z$: Vertical Elevation (Up)
