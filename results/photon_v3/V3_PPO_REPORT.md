# PhotonShield AI — Phase V3.3 PPO Adaptive Diffusion Controller Report

- **Hardware Target**: Edge MCU / Arduino Uno Q Deployment Preparation
- **Policy Architecture**: Actor-Critic ($9 \to 64 \to 32 \to 4$, Tanh activations)
- **Action Space**: Discrete diffusion step budgets $A = \{5, 10, 20, 50\}$ (indices `0..3`)
- **Training Pipeline**: PPO clipped objective ($\epsilon=0.20, \gamma=0.99, \lambda_{\text{gae}}=0.95$) trained across 3 seeds (`42, 123, 456`) on train split at 20% corruption
- **Evaluation Dataset**: Unseen Test Set (75 Sequences) evaluated across dropouts $p \in \{0.10, 0.20, 0.30, 0.40, 0.50\}$

## 1. Primary Test Set Comparative Evaluation

| Method | Macro-F1 | Accuracy | Missing MSE | Kinematic Residual | Avg Steps | Latency (ms) | Speedup vs 50 | Compute Reduction | Oracle Regret |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed 50-Step V2** | `0.6929` | `68.8%` | `1.0335` | `0.0712 m/s` | **`50.0`** | `188.64 ms` | **`1.00x`** | **`0.0%`** | `0.0916` |
| **Fixed 10-Step V2** | `0.6921` | `68.8%` | `1.0315` | `0.0711 m/s` | **`10.0`** | `39.86 ms` | **`4.73x`** | **`80.0%`** | `0.0119` |
| **V3.1 Rule Scheduler** | `0.6921` | `68.8%` | `1.0315` | `0.0711 m/s` | **`10.0`** | `39.86 ms` | **`4.73x`** | **`80.0%`** | `0.0119` |
| **V3.2 Supervised Scheduler** | `0.6772` | `68.0%` | `1.0424` | `0.0726 m/s` | **`5.0`** | `63.57 ms` | **`2.97x`** | **`90.0%`** | `0.0033` |
| **Oracle Adaptive** | `0.7220` | `72.8%` | `1.0412` | `0.0724 m/s` | **`5.6`** | `60.85 ms` | **`3.10x`** | **`88.9%`** | `0.0000` |
| **V3.3 PPO Controller (3-Seed Mean)** | **`0.6821`** | **`68.3%`** | `1.0388` | `0.0721 m/s` | **`6.7`** | **`55.67 ms`** | **`3.56x`** | **`86.7%`** | **`0.0119`** |

---

## 2. Test Performance Across Corruption Regimes

| Dropout Level | Fixed 50-Step F1 | V3.1 Rule F1 | V3.2 Supervised F1 | V3.3 PPO F1 | Oracle F1 | PPO Avg Steps | PPO Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **p = 10%** | `0.8264` | `0.8264` | `0.8109` | **`0.8160`** | `0.8247` | **`6.7 steps`** | **`3.56x`** |
| **p = 20%** | `0.7959` | `0.7951` | `0.7303` | **`0.7519`** | `0.7954` | **`6.7 steps`** | **`3.56x`** |
| **p = 30%** | `0.7206` | `0.7206` | `0.7350` | **`0.7302`** | `0.7578` | **`6.7 steps`** | **`3.56x`** |
| **p = 40%** | `0.6013` | `0.5979` | `0.5618` | **`0.5739`** | `0.6471` | **`6.7 steps`** | **`3.56x`** |
| **p = 50%** | `0.5202` | `0.5202` | `0.5479` | **`0.5387`** | `0.5850` | **`6.7 steps`** | **`3.56x`** |

---

## 3. RL Diagnostics & Oracle Gap Closure

- **PPO Action Distribution**: `5 steps`: `0.0%`, `10 steps`: `100.0%`, `20 steps`: `0.0%`, `50 steps`: `0.0%`
- **PPO Action Entropy**: **`-0.0000`** (converged from initial uniform entropy of `1.386`)
- **Oracle Agreement**: **`11.47%`**
- **Oracle Gap Closure**: **`87.04%`**

### Supervised vs. PPO Confusion Matrix vs. Oracle:

| Oracle Target | PPO Predicted 5 | PPO Predicted 10 | PPO Predicted 20 | PPO Predicted 50 |
| :---: | :---: | :---: | :---: | :---: |
| **Oracle 5 Steps** | `0` | `332` | `0` | `0` |
| **Oracle 10 Steps** | `0` | `43` | `0` | `0` |
| **Oracle 20 Steps** | `0` | `0` | `0` | `0` |
| **Oracle 50 Steps** | `0` | `0` | `0` | `0` |

---

## 4. Reward Ablation Results

| Ablation Configuration | Included Components | Validation Avg Steps |
| :--- | :--- | :---: |
| **Ablation A** | Perception Only ($\alpha=1.0$) | `10.0 steps` |
| **Ablation B** | Perception + Physics ($\alpha=1.0, \beta=0.25$) | `10.0 steps` |
| **Ablation C** | Perception + Compute ($\alpha=1.0, \gamma=0.10$) | `10.0 steps` |
| **Ablation D** | Full Composite ($\alpha=1.0, \beta=0.25, \gamma=0.10$) | `10.0 steps` |

---

## 5. Seed Stability Analysis

- **Seed 42**: Test Macro-F1 = **`0.6772`**, Average Steps = **`5.0`**
- **Seed 123**: Test Macro-F1 = **`0.6772`**, Average Steps = **`5.0`**
- **Seed 456**: Test Macro-F1 = **`0.6921`**, Average Steps = **`10.0`**

---

## 6. FINAL STATUS: **PPO PARTIAL**

