# PhotonShield AI -- Phase V5.5 Oxford Final Temporal-Physics Foundation Report

## 1. Scientific Research Questions
> **Primary Question**: *"Can a fully trained Mamba-based temporal radar representation, regularized by physical kinematic constraints, learn a robust temporal foundation from the complete Oxford radar dataset?"*

> **Secondary Question**: *"Does the resulting Oxford foundation provide a stable, transferable latent representation for downstream VoD 3D radar perception?"*

---

## 2. Sequence Length Ablation (Phase 4)

| Sequence Length | Reconstruction MSE | Kinematic Residual | Temporal Consistency | Sequence Latency (ms) | Parameters |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **T = 8** | `0.02011 ± 0.00015` | `0.01552` | `0.53586` | `10.65 ms` | `74,949` |
| **T = 16** | `0.02294 ± 0.00018` | `0.02358` | `0.58087` | `15.89 ms` | `74,949` |
| **T = 32** | `0.03520 ± 0.00135` | `0.03716` | `0.63880` | `31.14 ms` | `74,949` |

---

## 3. Physics Regularization Ablation (Phase 6)

| Physics Weight (\lambda_{\text{phys}}) | Reconstruction MSE | Kinematic Residual |
| :---: | :---: | :---: |
| **\lambda = 0.00** | `0.02294 ± 0.00018` | `0.53844 ± 0.04824` |
| **\lambda = 0.01** | `0.02294 ± 0.00018` | `0.02358 ± 0.00240` |
| **\lambda = 0.05** | `0.02293 ± 0.00018` | `0.02335 ± 0.00237` |

---

## 4. Robustness & Corruption Benchmark (Phase 5)

| Corruption Regime | Reconstruction MSE | Missing Frame MSE | Kinematic Residual |
| :--- | :---: | :---: | :---: |
| **Clean (p=0%)** | `0.02269` | `0.00000` | `0.02054` |
| **Bernoulli p=0.10** | `0.03609` | `0.14219` | `0.02225` |
| **Bernoulli p=0.20** | `0.04138` | `0.10195` | `0.02439` |
| **Bernoulli p=0.30** | `0.05940` | `0.13289` | `0.02613` |
| **Bernoulli p=0.40** | `0.07702` | `0.14242` | `0.02899` |
| **Bernoulli p=0.50** | `0.08036` | `0.13559` | `0.03163` |
| **Contiguous Gap G=2** | `0.02412` | `0.03211` | `0.02248` |
| **Contiguous Gap G=4** | `0.02778` | `0.04130` | `0.02525` |
| **Contiguous Gap G=8** | `0.05205` | `0.07945` | `0.03714` |

---

## 5. Latent Representation & Semantic Manifold (Phase 11)

- **Latent Mean**: `0.1458`
- **Latent Standard Deviation**: `1.5215`
- **Latent Min / Max**: `[-2.882, +7.787]`
- **Mean Absolute Value**: `1.1627`
- **Temporal Smoothness**: `0.5704`

---

## 6. FP32 Deployment Footprint Audit (Phase 10)

- **Total Trainable Parameters**: `74,949`
- **FP32 Weight Memory**: `0.29 MB`
- **Single-Sequence Inference Latency**: `16.44 ms` (60.8 FPS)
- **Compute FLOPs per Sequence**: `0.14 MFLOPs`

---

## 7. Scientific Conclusion & Status

> **FINAL STATUS: `V5.5 OXFORD FOUNDATION READY`**

- **Permanent Checkpoint**: `checkpoints/v5_5/oxford_final/oxford_final_foundation.pt`
- **Recommended Next Step**: Frozen/Fine-tuned transfer to VoD 3D object perception.
