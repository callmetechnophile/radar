# UNO Q Benchmark Plan

The V9 model has already been characterized at model level. This directory is for physical UNO Q measurements only.

## Required measurements

1. FP32 reference.
2. INT8 weight-only reference.
3. Single-frame latency.
4. T16 latency.
5. Sustained 30 Hz throughput.
6. QRB2210 CPU utilization.
7. Resident memory / peak memory.
8. IPC overhead between STM32U585 and QRB2210.
9. Thermal behavior over a sustained workload.
10. Power consumption.

## Acceptance

A deployment result is only marked PASS after measurements are captured on the actual board. Clock-based estimates are not accepted as hardware evidence.
