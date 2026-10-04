# Hardware Target Matrix

| Target | Compute | Memory | Role | Status |
|---|---|---|---|---|
| ESP32-S3 | Xtensa LX7 MCU | 8 MB PSRAM variant supported | lightweight acquisition/control/transport test | planned |
| UNO Q QRB2210 4GB | 4x Cortex-A53 + Adreno 702 | 4 GB LPDDR4 + 32 GB eMMC | W-band DSP + PhotonNet inference | preferred |
| UNO Q QRB2210 2GB | 4x Cortex-A53 + Adreno 702 | 2 GB LPDDR4 + 16 GB eMMC | W-band DSP + PhotonNet inference | supported |
| UNO Q STM32U585 | Cortex-M33 @ up to 160 MHz | 786 KB SRAM + 2 MB Flash | deterministic I/O, timing, watchdog | control domain |

## Rule

One canonical V9 model. Hardware-specific runtime adapters only. Do not create separate neural models for ESP32-S3 and UNO Q.

## Deployment order

1. Validate canonical FP32 V9.
2. Validate INT8 weight-only V9.
3. Benchmark ESP32-S3 for the subset it can realistically execute; do not force full inference.
4. Benchmark QRB2210 4GB for full inference.
5. Validate STM32U585 acquisition/control and IPC.
6. Measure end-to-end 30 Hz operation.
