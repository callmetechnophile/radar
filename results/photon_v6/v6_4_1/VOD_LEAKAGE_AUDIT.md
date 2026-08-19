# View-of-Delft (VoD) Dataset Isolation & Leakage Audit

## 1. Split Intersection Verification

- **Train ∩ Validation Overlap**: `0` frames (Zero Leakage Verified)
- **Train ∩ Test Overlap**: `0` frames (Zero Leakage Verified)
- **Validation ∩ Test Overlap**: `0` frames (Zero Leakage Verified)
- **Total Unique Frames in Manifest**: `8,682` frames

## 2. Sequence Boundary Isolation

- Temporal windows are strictly partitioned within continuous driving sequences.
- Zero temporal windows cross between train, validation, or test partitions.
- Normalization statistics are computed exclusively on the 5,139 training frames.
