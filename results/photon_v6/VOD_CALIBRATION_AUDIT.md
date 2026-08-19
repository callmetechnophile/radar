# View-of-Delft (VoD) Sensor Calibration & Coordinate Transformation Audit

## 1. Camera Projection Matrix $P_2$ (3x4)
```
[[1.49546864e+03 0.00000000e+00 9.61272442e+02 0.00000000e+00]
 [0.00000000e+00 1.49546864e+03 6.24895920e+02 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00 0.00000000e+00]]
```

## 2. Radar to Camera Rigid Extrinsics $T_{\text{cam}\leftarrow\text{radar}}$ (3x4)
```
[[-0.013857   -0.9997468   0.01772762  0.05283124]
 [ 0.10934269 -0.01913807 -0.99381983  0.98100483]
 [ 0.99390751 -0.01183297  0.1095802   1.44445002]]
```
- Rotation Orthogonality $R^\top R \approx I$: Maximum deviation = `5.675587e-08`
- Determinant $\det(R)$: `1.000000` (Expected $+1.000000$ -> **Valid SO(3)**)

## 3. LiDAR to Camera Rigid Extrinsics $T_{\text{cam}\leftarrow\text{lidar}}$ (3x4)
```
[[-0.0079802 -0.9998541  0.0151049  0.151    ]
 [ 0.118497  -0.0159445 -0.9928264 -0.461    ]
 [ 0.9929224 -0.0061331  0.1186069 -0.915    ]]
```
- Rotation Orthogonality $R^\top R \approx I$: Maximum deviation = `1.150228e-07`
- Determinant $\det(R)$: `1.000000` (Expected $+1.000000$ -> **Valid SO(3)**)
