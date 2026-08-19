# View-of-Delft (VoD) Radar Field Semantics & Physical Definitions

- **Sensor Model**: ZF 3D Full-Range Radar (77 GHz FMCW, 192 virtual channels, elevation + azimuth)
- **Point-Cloud Array Shape**: `(N, 7)` of type `float32`
- **Native Coordinate System**: ISO 8855 Radar Reference Frame ($+X$ Forward, $+Y$ Left, $+Z$ Up)

## Verified Field Table

| Index | Field Name | Physical Meaning | Unit | Coordinate Frame | Empirical Range (Min .. Max) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| `0` | **`x`** | Longitudinal distance along sensor forward axis | m | Radar | `-0.27 .. 86.77 m` |
| `1` | **`y`** | Lateral displacement (left = +y, right = -y) | m | Radar | `-47.61 .. 79.49 m` |
| `2` | **`z`** | Vertical elevation (up = +z, down = -z) | m | Radar | `-17.65 .. 15.56 m` |
| `3` | **`RCS`** | Radar Cross Section / Reflection Power | dBsm | Radar Antenna | `-66.55 .. 55.51 dBsm` |
| `4` | **`v_r`** | Raw radial Doppler velocity (receding = +, approaching = -) | m/s | Radar Radial Beam | `-24.36 .. 24.77 m/s` |
| `5` | **`v_r_compensated`** | Ego-motion compensated target Doppler velocity | m/s | Vehicle Frame | `-24.80 .. 26.46 m/s` |
| `6` | **`time_id`** | Relative frame index in multi-scan accumulation | frames | Temporal | `0.0 .. 0.0` |
