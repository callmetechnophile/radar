# Oxford Radar RobotCar Final Data Foundation Audit (Phase V5.5)

## 1. Dataset Scale & Representation Summary

| Dataset Attribute | Audited Measurement | Validation Verdict |
| :--- | :--- | :--- |
| **Dataset Source** | Oxford Radar RobotCar (`2019-01-10-14-36-48-radar-oxford-10k-partial`) | Verified on Local Filesystem |
| **Total Radar Scans** | **`9,022` complete 360° scans** | 100% Valid PNG Scans |
| **Radar Representation** | Polar Grid: `400 azimuths × 3768 range bins` | Native Navtech CTS350-X format |
| **Range Resolution** | $0.0432\text{ m / bin}$ ($162.78\text{ m}$ maximum range) | Calibrated |
| **Azimuth Resolution** | $0.9^\circ$ per beam ($400$ beams per rotation) | Nominal |
| **Temporal Frequency** | $4.0\text{ Hz}$ ($\Delta t = 250\text{ ms}$ inter-frame spacing) | Continuous monotonic timestamps |
| **Missing / Corrupted Frames** | **0 frames** (zero NaN, zero Inf, zero decode failures) | Verified Complete |
| **Odometry / Pose Sources** | GPS/INS Ground Truth + Visual Odometry (VO) | Synchronized |

---

## 2. Sequence-Level Partition Protocol

To prevent temporal leakage across sequential radar scans, sequences are partitioned strictly into contiguous temporal chunks:

| Split Partition | Percentage | Scan Range | Total Scans | Sequence Count ($T=8$) | Sequence Count ($T=16$) | Sequence Count ($T=32$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | 70.0% | `000000 .. 006314` | **`6,315` scans** | 789 sequences | 394 sequences | 197 sequences |
| **Validation Split** | 15.0% | `006315 .. 007667` | **`1,353` scans** | 169 sequences | 84 sequences | 42 sequences |
| **Testing Split** | 15.0% | `007668 .. 009021` | **`1,354` scans** | 169 sequences | 84 sequences | 42 sequences |

---

## 3. Training Set Normalization Statistics
*Computed strictly on the 6,315 training scans (zero test-set leakage):*
- **Feature Channel Mean**: `0.1142`
- **Feature Channel Standard Deviation**: `0.1874`
- **Min / Max Values**: `[0.0000, 1.0000]`
