# PhotonShield AI -- Phase V6.1 Repository Integration Document

## 1. Architectural Foundation & Component Reuse
Phase V6.1 builds directly upon the validated mathematical foundations established in V4 and V5 while introducing native 3D spatial perception:

| Component | Source in Repository | Role in Phase V6.1 |
| :--- | :--- | :--- |
| **`MiniMambaBlock`** | [`module_04_mamba_hybrid/mamba_core.py`](file:///C:/Users/worka/research/photonpinn/radar/module_04_mamba_hybrid/mamba_core.py) | Reused as the core causal selective state-space temporal recurrence block ($d_{\text{model}}=64, d_{\text{state}}=16, d_{\text{conv}}=4$) |
| **`OxfordMambaTemporalModel`** | [`module_07_temporal/mamba_temporal.py`](file:///C:/Users/worka/research/photonpinn/radar/module_07_temporal/mamba_temporal.py) | Reused architecture template for temporal feature sequence modeling $[B, T, 64]$ |
| **Seed & Determinism** | `torch.manual_seed`, `np.random.seed` | Exact 3-seed protocol (`42, 123, 456`) |
| **Edge Footprint Audit** | [`experiments/run_v4_fp32_audit.py`](file:///C:/Users/worka/research/photonpinn/radar/experiments/run_v4_fp32_audit.py) | Param count, FLOPs, Latency, Peak RAM/VRAM benchmarking methodology |

---

## 2. VoD Native Radar Integration Layer (`module_08_vod/`)
- **Native Radar Loader**: Ingests un-aggregated $(N, 7)$ single-scan radar point clouds ($[x, y, z, \text{RCS}, v_r, v_{r,\text{comp}}, \text{time\_id}]$).
- **Point Encoder**: Shared MLP $(7 \to 32 \to 64)$ with permutation-invariant max-pooling across points $N \to 64$-D frame embedding.
- **Sequence Builder**: Non-overlapping sliding temporal sequences of length $T \in \{8, 16\}$ extracted strictly within continuous driving snippets.
- **3D Occupancy Supervision**: Synchronized LiDAR point clouds transformed to the radar coordinate frame and voxelized into a $32 \times 32 \times 8$ grid ($8,192$ cells) over $X \in [0, 32]\text{ m}, Y \in [-16, 16]\text{ m}, Z \in [-2.5, 2.5]\text{ m}$.
- **Decoders**:
  - Baseline A (Frame-wise): $64 \to 128 \to 512 \to 8192$ (No temporal modeling)
  - Baseline B (Mamba): $64 \to \text{Mamba}(64 \to 64) \to 128 \to 512 \to 8192$ (Temporal selective SSM)

---

## 3. Strict Scientific Non-Interference Rules
1. **No LiDAR Input**: LiDAR is strictly used as target ground-truth occupancy supervision for loss evaluation.
2. **No Pre-Accumulation**: Only native `radar/` single scans are used for the primary comparison.
3. **No Sequence Boundary Leakage**: Sequences never cross continuous driving snippet boundaries.
4. **Frozen Historical Foundations**: V1, V2, V3.1, V4, V5.0, V5.4 remain untouched.
