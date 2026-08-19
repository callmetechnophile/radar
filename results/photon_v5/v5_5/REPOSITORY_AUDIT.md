# PhotonShield AI — Phase V5.5 Repository & Architecture Audit

## 1. Architectural Components Audit

| Component | Repository Path | Frozen Status | Reuse Strategy |
| :--- | :--- | :--- | :--- |
| **Mamba Core** | `module_04_mamba_hybrid/mamba_core.py` | Validated & Frozen | Reused as base selective state space block (`MiniMambaBlock`, $d_{\text{state}}=16, d_{\text{conv}}=4$) |
| **Oxford Adapter** | `module_07_temporal/oxford_adapter.py` | Validated & Frozen | Ingests native Oxford polar scans `(400, 3768)`, timestamps, and odometry |
| **Feature Extractor** | `module_07_temporal/feature_extractor.py` | Validated & Frozen | Maps polar scans to $64$-D compact temporal features |
| **Physics Head** | `module_07_temporal/physics_mamba.py` | Validated & Frozen | Auxiliary kinematics head predicting $[\Delta x, \Delta y, v_x, v_y, \omega]$ ($64 \to 32 \to 5$) |
| **Physics Losses** | `module_06_physics/physics_losses.py` | Validated & Frozen | Differentiable kinematic consistency & soft bounded acceleration penalty |
| **Sequence Builder** | `module_07_temporal/temporal_sequence.py` | Validated & Frozen | Builds non-overlapping sequential windows ($T=8, 16, 32$) partitioned strictly by drive |
| **Corruption Suite** | `module_07_temporal/temporal_corruption.py` | Validated & Frozen | Evaluates Bernoulli dropout ($p \in [0.1, 0.5]$) and contiguous gaps ($G \in [2, 8]$) |

---

## 2. Checkpoint Selection & Evaluation Protocol
- **Checkpoint Policy (Policy B)**: 3-epoch smoothed validation loss with a 5-epoch warmup period. Prevents transient overfitting and ensures monotonic convergence.
- **Multi-Seed Protocol**: Strict 3-seed evaluation (`42, 123, 456`) reporting mean $\pm$ standard deviation across all metrics.
- **Inference Guarantee**: Radar only. Physical observables are strictly auxiliary regularization during training.
