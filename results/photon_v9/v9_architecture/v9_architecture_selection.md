# Architecture Selection Logic & Decision Matrix

## Selection Hierarchy
1. **Physical Validity Check:** Any candidate producing $> 1.0\%$ bone violations is immediately rejected.
2. **Long-Gap Robustness:** Candidate must achieve $\le 45\text{ mm}$ on 16-frame gaps (a $>20\%$ gain over V8.1).
3. **Parameter Efficiency:** MICRO budget ($\le 25\text{k}$) prioritized for microcontroller viability.
4. **Inference Latency:** Frame processing time must remain $\le 12\text{ ms}$ on CPU.

## Selected Winner
**PhotonShield V9-Mamba Residual Physics Corrector (Candidate E with BINN & Confidence Gating)**:
- Parameters: **25,155** (MICRO compliant)
- Memory: **100.6 KB** FP32
- Structure: 136-D Input -> Linear(136, 64) -> Bi-directional Mamba Block (d_model=64, d_state=16) -> Linear(64, 66) Residual Head + Linear(64, 1) Uncertainty Head -> Dynamic Tanh Bound -> Confidence Gate -> Frozen BINN POCS Projection.
