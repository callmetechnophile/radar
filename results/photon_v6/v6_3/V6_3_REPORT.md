# PhotonShield AI -- Phase V6.3 Full-Scale VoD 3D Perception & Multi-Object Tracking Report

## 1. Scientific Objectives & Hypotheses
> **Primary Question**: *"Does the validated V6.2 radar-temporal-physics representation retain its advantage when trained on the complete VoD training dataset?"*

> **Secondary Question**: *"Which lightweight multi-object prediction head (Anchor-based vs Query-based) is most suitable for dense VoD radar scenes?"*

---

## 2. Multi-Object Head Comparison (Stage B)

| Architecture Head | 3D mAP | BEV mAP | Center MAE (m) | Class Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **HEAD-1 (Anchor-Based)** | `0.0044 ± 0.0011` | `0.0072 ± 0.0017` | `6.527 m` | `0.2843` |
| **HEAD-2 (Query-Based)** | `0.0056 ± 0.0026` | `0.0087 ± 0.0044` | `5.193 m` | `0.2576` |

---

## 3. Physics Regularization Ablation (Stage A)

| Physics Regularization | 3D mAP | BEV mAP | Center MAE (m) | Class Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Physics lambda=0.00** | `0.0057 ± 0.0010` | `0.0096 ± 0.0030` | `5.008 m` | `0.2316` |
| **Physics lambda=0.01** | `0.0056 ± 0.0026` | `0.0087 ± 0.0044` | `5.193 m` | `0.2576` |
| **Physics lambda=0.05** | `0.0076 ± 0.0012` | `0.0120 ± 0.0017` | `5.260 m` | `0.2402` |

---

## 4. Multi-Object Tracking Benchmark (Stage D)

- **HOTA (Higher Order Tracking Accuracy)**: `0.2781`
- **IDF1 (ID F1-Score)**: `0.1436`
- **MOTA (Multi-Object Tracking Accuracy)**: `0.0000`
- **ID Switches**: `0`
- **Track Fragmentations**: `0`
- **Mean Trajectory Position Error**: `1.616 m`

---

## 5. Edge Deployment Footprint Audit

- **Total Trainable Parameters**: `4,377,019`
- **Weight Memory (FP32)**: `16.70 MB`
- **Sequence Latency (GPU)**: `12.69 ms` (78.8 FPS)
- **Compute FLOPs per Sequence**: `8.72 MFLOPs`

---

## 6. Scientific Verdict

> **STATUS: `V6.3 FULL-SCALE 3D PERCEPTION & TRACKING COMPLETE`**
