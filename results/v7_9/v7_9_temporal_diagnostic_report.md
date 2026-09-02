# PhotonShield AI — Phase V7.9 Temporal Failure Diagnostic Report

## Executive Summary
This diagnostic investigation isolates the precise mechanism causing catastrophic pose degradation under temporal corruption in the frozen M4Human foundation.

---

## 1. Primary Diagnostic Findings

1. **Clean Reproduction: PASS**
   - Baseline MPJPE: `60.9 mm` (Expected ~61.1 mm)
   - BINN MPJPE: `77.7 mm` (Expected ~77.7 mm)

2. **Missing-Frame Representation Ablation (Section 5):**
   - Zero-replacement: `308.5 mm`
   - Previous-frame: `61.4 mm`
   - Last-valid ZOH: `61.4 mm`
   - Linear Interpolation: `61.0 mm`
   - Mean-token: `240.7 mm`
   *Crucial finding*: Simply holding the last valid frame or interpolating reduces gap degradation from `308.5 mm` down to `61.0 mm` without retraining.

3. **Mamba State Contamination (Section 7):**
   - The Mamba SSM updates state via `h_t = h_{t-1} * dA + dB * u_t`.
   - When an unobserved frame is represented as zeros, `u_t` delivers a sudden impulse shock that abruptly contaminates `h_t`.
   - Because Mamba is bidirectional, this error propagates both forward and backward, corrupting the entire 16-frame window.

4. **Pose Head Stability (Section 8):**
   - The pose head is NOT inherently unstable. When fed clean latents, pose decoding is stable. The instability originates upstream in the recurrent Mamba representation.

5. **Primary Failure Category:**
   - **`H` (Multiple Interacting Causes)**
      - 1. Category C: Mamba bidirectional state contamination from zero-token impulse shock
   - 2. Category B: Out-of-distribution representation shift (zero feature vector vs continuous training distribution)
   - 3. Category F: Absence of observation-mask gating inside the SSM recurrence (Mamba treats zero as an active measurement)

---

## 2. Recommended Next Experiment
> **Observation-Gated Mamba SSM (carry-forward state when mask=0, bypassing zero-token update)**
