# Temporal Corruption Implementation Specification

1. Frame Removal Mechanism:
   - Evaluates Bernoulli dropout: each frame t is independently dropped with probability p.
   - Evaluates Contiguous gaps: consecutive frames [start, start + gap] are dropped.
2. Representation of Missing Frames:
   - In raw historical implementation (V7.6): missing frames were replaced by torch.zeros([B, T, 64]).
   - Bug Identified: When zero tokens are passed through extract_radar_domain_descriptor, standard deviation underflow occurs on near-constant features (e.g. std[5]=7.67e-7), resulting in an artificial -300,000 normalized descriptor spike and a false 72,000+ mm offset.
3. Verified Correction:
   - In mask-aware safe descriptor extraction, sequence-level statistics are computed exclusively over observed frames, and missing frames use zero-order hold (ZOH) from nearest observed frames.
4. Normalization Timing:
   - Radar token encoding occurs prior to sequence window assembly.
   - Static adapter normalization is applied per-sequence during inference.
5. Mamba State Dynamics:
   - Mamba hidden state h_t starts at 0 at t=0 for each window and propagates recurrently forward and backward across T=16.
   - When a frame has 0 features, the state transition equation h_t = h_{t-1} * dA + dB * u receives an abrupt discontinuous shock.
6. Determinism:
   - Fully deterministic under fixed NumPy / PyTorch seed.
