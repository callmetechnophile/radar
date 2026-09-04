# V9 Layerwise Tensor Dimension & Shape Audit

## 1. Sequence Tensor Flow (T=16 Temporal Preservation)
The architecture operates strictly on sequences of shape `[B, T, D]` without flattening the temporal dimension before sequence modeling.

| Processing Stage | Module / Operation | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Input Interface** | `Raw Observation Window` | `[B, T, 136]` | `[B, T, 136]` | Unflattened temporal sequence |
| **Stem Embedding** | `nn.Linear(136, 64, bias=False)` | `[B, T, 136]` | `[B, T, 64]` | Pointwise projection along D |
| **Normalization** | `nn.LayerNorm(64)` | `[B, T, 64]` | `[B, T, 64]` | Channel normalization |
| **Activation** | `nn.SiLU()` | `[B, T, 64]` | `[B, T, 64]` | Non-linear gating stem |
| **SSM In-Projection**| `nn.Linear(64, 2*d_inner)` | `[B, T, 64]` | `[B, T, 94]` | Splits to x and z branches |
| **Causal 1D Conv** | `nn.Conv1d(47, 47, k=3, g=47)` | `[B, 47, T]` | `[B, 47, T]` | Depthwise causal convolution |
| **Selective SSM** | `Discretized S6 Recurrence` | `[B, T, 47]` | `[B, T, 47]` | State space tracking |
| **SSM Out-Projection**| `nn.Linear(47, 64, bias=False)` | `[B, T, 47]` | `[B, T, 64]` | Channel expansion back to 64 |
| **Residual Head** | `nn.Linear(64, 66, bias=False)` | `[B, T, 64]` | `[B, T, 66]` | 22 joints x 3 coordinates |
| **Uncertainty Head**| `nn.Linear(64, 1, bias=True)` | `[B, T, 64]` | `[B, T, 1]` | Residual epistemic dispersion |
| **Dynamic Bounding**| `r_max * tanh(r / r_max)` | `[B, T, 66]` | `[B, T, 66]` | Dynamic spatial saturation |
| **Confidence Gate** | `alpha_t * bounded_r` | `[B, T, 66]` | `[B, T, 66]` | Signal attenuation |
| **Pose Addition** | `y_math + gated_r` | `[B, T, 66]` | `[B, T, 66]` | Reconstructed 3D pose |
| **POCS Projection** | `FK Tree + Rodrigues Cones` | `[B, T, 22, 3]` | `[B, T, 22, 3]` | Guaranteed 0.0% violations |

## 2. Conclusion
- Sequence shape is strictly preserved across all layers: **PASS**
- No premature temporal collapsing or flattening: **PASS**
