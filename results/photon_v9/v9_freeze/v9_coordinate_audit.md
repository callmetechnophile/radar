# V9 Coordinate System & Reference Frame Audit

## 1. Frame Definition
- **Reference Standard**: M4Human / ISO 8855 Right-Handed Cartesian Coordinate System.
- **X-axis**: Lateral / Transverse (+X points to the subject's right when facing forward, or camera/radar right).
- **Y-axis**: Longitudinal / Forward (+Y points forward along range direction).
- **Z-axis**: Vertical (+Z points upward against gravity).
- **Handedness**: Right-Handed: $\mathbf{\hat{x}} \times \mathbf{\hat{y}} = \mathbf{\hat{z}}$.
- **Units**: Meters (m) strictly.

## 2. Dataset Alignment Analysis
| Framework | Coordinate Convention | Scale / Operating Range | Root Origin | Handedness |
| :--- | :--- | :--- | :--- | :--- |
| **M4Human (MoCap GT)** | $+X$ right, $+Y$ forward, $+Z$ up | Indoor ($2.0 - 6.0\text{ m}$) | Ground/Pelvis center | Right-Handed |
| **VoD Foundation (V6.4)**| $+X$ right, $+Y$ forward, $+Z$ up | Automotive ($0.0 - 32.0\text{ m}$) | Front bumper center | Right-Handed |
| **V8.1 Canonical Model** | $+X$ right, $+Y$ forward, $+Z$ up | Metric ($2.0 - 6.0\text{ m}$) | Pelvis (Root joint 0) | Right-Handed |
| **V9 Residual Target** | $+X$ right, $+Y$ forward, $+Z$ up | Metric Discrepancy (m) | Zero-centered on V8.1 | Right-Handed |

## 3. VoD to M4Human Transfer Offset Resolution
In Phase V7.2 and V7.3, transferring the automotive VoD foundation directly to M4Human introduced an apparent coordinate translation error (MPJPE = $95.9\text{ mm}$ with $2.4\text{ m}$ range offset). This was proven in V7.3 to be an affine coordinate origin difference rather than an articulated body geometry error.
V8.1 and V9 resolve this completely by operating directly in the calibrated M4Human metric frame.

## 4. Verification Conclusion
- Coordinate convention: **PASS**
- Units: **PASS (Meters)**
- Axis alignment: **PASS (+X right, +Y forward, +Z up)**
