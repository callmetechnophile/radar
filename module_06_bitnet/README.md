# Module 6 — Low-Bit Runtime and Edge Hardware

Module 6 contains low-bit experiments and deployment accounting for the current V9 PhotonNet architecture.

## Current canonical hardware targets
- Arduino UNO Q QRB2210 4 GB / 32 GB — preferred inference host.
- Arduino UNO Q QRB2210 2 GB / 16 GB — supported inference host.
- Arduino UNO Q STM32U585 — deterministic acquisition/control domain.
- ESP32-S3 — separate lightweight hardware test target; not the primary V9 inference host.

The UNO Q must not be represented as a single 512 KB/64 KB MCU. Those values were a legacy profile and have been removed.

## V9 low-bit reference

| Variant | Clean MPJPE | 16-frame gap | Weight size | Peak RAM |
|---|---:|---:|---:|---:|
| FP32 | 93.77 mm | 116.05 mm | 96.79 KB | 140.66 KB |
| FP16 | 93.77 mm | 116.05 mm | 48.39 KB | 70.33 KB |
| INT8 weight-only | 93.77 mm | 116.06 mm | 24.20 KB | 68.07 KB |
| INT8 dynamic | 95.54 mm | 117.83 mm | 24.20 KB | 37.37 KB |
| INT8 static | 95.55 mm | 117.84 mm | 24.20 KB | 37.37 KB |
| INT4 weight-only | 93.86 mm | 116.14 mm | 12.10 KB | 55.97 KB |

These are model-level measurements, not UNO Q hardware benchmarks.

### Preferred reference
Use INT8 weight-only as the current accuracy-preserving deployment candidate. Do not claim real-world speedup until measured on QRB2210.

## Legacy code policy
Files or scripts that instantiate PhotonV0 are historical/legacy unless a research experiment explicitly targets that architecture. New deployment code must use the frozen V9 interface and its 136-D input / 66-D residual output.

MNN is not part of the deployment stack.