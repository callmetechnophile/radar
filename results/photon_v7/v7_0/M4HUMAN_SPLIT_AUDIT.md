# M4Human Official Dataset Split & Leakage Audit

## Dataset Population
- Total Subjects: 20 human subjects (P1 to P20, 8 male, 12 female)
- Total Actions: 50 distinct motor actions (rehabilitation, sports, daily living)
- Total Sequences: 1,000 continuous action sequences
- Total Frames: ~661,000 multimodal frames at 30.0 Hz

## Benchmark Split Schemes
1. Scheme S2: Cross-Subject Split (Primary Protocol)
   - Training Set: 15 subjects (P1, P2, P3, P4, P6, P7, P9, P11, P12, P13, P14, P15, P16, P18, P20) -> 75.0% of sequences (750 sequences, ~495,750 frames)
   - Validation Set: 1 subject (P17) -> 5.0% of sequences (50 sequences, ~33,050 frames)
   - Test Set: 4 subjects (P5, P8, P10, P19) -> 20.0% of sequences (200 sequences, ~132,200 frames)
   - Leakage Status: ZERO SUBJECT OVERLAP. Complete actor generalization.

2. Scheme S3: Cross-Action Split (Secondary Protocol)
   - Training Set: 37 unseen actions -> 74.0%
   - Validation Set: 3 unseen actions (A10, A24, A32) -> 6.0%
   - Test Set: 10 unseen actions (A2, A11, A23, A28, A29, A33, A38, A43, A47, A50) -> 20.0%
   - Leakage Status: ZERO ACTION OVERLAP. Complete motion class generalization.

3. Scheme S1: Random Split (Baseline Protocol)
   - Split per sequence: 75% train, 5% val, 20% test.

## Temporal Window Capacity (No Boundary Crossing)
- Under T=16 continuous temporal windows:
  - Train Windows (S2): ~484,500 sequences
  - Validation Windows (S2): ~32,300 sequences
  - Test Windows (S2): ~129,200 sequences
  - Cross-Sequence Window Leakage: 0.00% (Guaranteed Zero Boundary Crossing).
