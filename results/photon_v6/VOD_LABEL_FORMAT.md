# View-of-Delft (VoD) 3D Bounding-Box Label Format & Class Inventory

- **Coordinate System**: KITTI Camera 2 Coordinates ($+X$ Right, $+Y$ Down, $+Z$ Forward along optical axis)
- **Total Annotated Frames**: `6,435` | **Total 3D Bounding Boxes**: `50,568`
- **Mean Objects Per Frame**: `7.86`

## KITTI Line Field Specification (16 fields):
1. `type`: Object class name (`Car`, `Pedestrian`, `Cyclist`, `rider`, `truck`, `bus`, `motor`, etc.)
2. `truncation`: Float from 0 (non-truncated) to 1 (truncated across image boundary)
3. `occlusion`: Integer (0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown)
4. `alpha`: Observation angle $\alpha \in [-\pi, \pi]$
5. `bbox_2d`: 4 floats $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ in image pixel coordinates
6. `dimensions_3d`: 3 floats $[h, w, l]$ (height, width, length in meters)
7. `location_3d`: 3 floats $[x, y, z]$ (bottom-center position in camera frame in meters)
8. `rotation_y`: Rotation around camera vertical axis $Y_{\text{cam}}$ in $[-\pi, \pi]$
9. `score`: Confidence / presence indicator

## Verified Class Distribution Table

| Class Name | Total 3D Objects | Share (%) |
| :--- | :---: | :---: |
| **`Car`** | `19,899` | `21.74%` |
| **`Pedestrian`** | `19,892` | `21.73%` |
| **`bicycle`** | `17,324` | `18.93%` |
| **`bicycle_rack`** | `10,195` | `11.14%` |
| **`rider`** | `9,508` | `10.39%` |
| **`Cyclist`** | `8,119` | `8.87%` |
| **`moped_scooter`** | `3,702` | `4.04%` |
| **`ride_other`** | `1,265` | `1.38%` |
| **`motor`** | `571` | `0.62%` |
| **`human_depiction`** | `370` | `0.40%` |
| **`vehicle_other`** | `290` | `0.32%` |
| **`truck`** | `219` | `0.24%` |
| **`DontCare`** | `101` | `0.11%` |
| **`ride_uncertain`** | `70` | `0.08%` |
