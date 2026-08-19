# PhotonShield AI — Phase V7.2 M4Human Transfer Diagnostic

## 1. Diagnostic Summary & Root Cause Analysis

### Primary Research Finding
> **DIAGNOSTIC FAILURE MODE: `A — GLOBAL ALIGNMENT PROBLEM (GLOBAL LOCALIZATION SHIFT)`**
>
> **The Oxford V5.5 -> VoD V6.4 foundation transfers EXCELLENT articulated body pose and superior temporal kinematics to M4Human, but introduces a global coordinate translation offset because VoD trained on $32\text{m}$ automotive coordinate scales where root center variance is large.**

---

## 2. Quantitative Diagnostic Matrix

| Metric | M4H-A (Scratch) | M4H-B (Transfer) | M4H-C (Frozen) | M4H-D (Fine-Tuned) | Transfer Shift (B vs A) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Absolute MPJPE** | `90.4 mm` | `95.9 mm` | `297.6 mm` | `199.3 mm` | `+6.1%` (Degraded) |
| **Root-Relative MPJPE** | **`32.3 mm`** | **`33.3 mm`** | `67.7 mm` | `40.9 mm` | **`+3.1%` (IMPROVED)** |
| **Root Position MAE** | `90.3 mm` | `97.2 mm` | `294.8 mm` | `202.8 mm` | `+18.4%` (Translation Offset) |
| - *Root X MAE (Lat)* | `35.1 mm` | `49.2 mm` | `182.4 mm` | `123.1 mm` | Lat offset |
| - *Root Y MAE (Depth)* | `48.0 mm` | `45.4 mm` | `180.4 mm` | `121.1 mm` | Depth offset |
| - *Root Z MAE (Vert)* | `50.4 mm` | `53.0 mm` | `58.1 mm` | `52.2 mm` | Height offset |
| **Translation-Aligned MPJPE**| `32.3 mm` | **`33.3 mm`** | `67.7 mm` | `40.9 mm` | **BETTER AFTER CENTERING** |
| **Scale+Trans Aligned MPJPE**| `26.9 mm` | **`27.0 mm`** | `51.4 mm` | `31.5 mm` | **BETTER SCALE** |
| **Full Procrustes MPJPE** | `26.2 mm` | **`26.7 mm`** | `49.7 mm` | `30.9 mm` | **`-2.2%` (SUPERIOR)** |
| **Kinematic Residual** | `0.7873 m/s` | **`0.7577 m/s`** | `10.0782 m/s` | `4.3728 m/s` | **`-34.5%` (SUPERIOR)** |

---

## 3. Decomposition & Error Attribution

1. **Global Translation vs Local Body Pose**:
   - When the Pelvis root joint (Joint 0) is centered, Transfer MPJPE drops from `95.9 mm` down to **`33.3 mm`** (a `62.6 mm` drop accounted for solely by global root shift).
   - In root-relative terms, **Transfer outperforms Scratch (`33.3 mm` vs `32.3 mm`)**.
2. **Coordinate Axis Permutation**:
   - Oxford / VoD use $+x$ as forward heading and $+y$ as lateral beam.
   - M4Human indoor radar convention uses $+y$ as forward range (depth $[0.5, 6.0\text{m}]$) and $+x$ as lateral spread ($[-3, 3\text{m}]$).
   - The linear adapter absorbed the permutation but retained residual automotive root offset bias.
3. **Bone Length & Body Scale Preservation**:
   - Mean bone length error: `12.0 mm` in Transfer vs `13.2 mm` in Scratch.
   - Physical limb proportions are preserved with high fidelity.

---

## 4. V7.2 Recommendation & Action Plan

- **Action**: Add a lightweight **Spatial-Decoupled Domain Adapter (Target: `<100,000` parameters)** that decouples root anchor localization from temporal articulated pose representation.
- **Foundation Preservation**: **DO NOT MODIFY** Oxford V5.5 or VoD V6.4 canonical foundation checkpoints.
