# Arduino UNO Q Hardware Target

## Heterogeneous architecture

The UNO Q is treated as two cooperating compute domains:

| Domain | Hardware | Primary role |
|---|---|---|
| Linux/application | QRB2210, 4x Cortex-A53 up to 2 GHz, Adreno 702, 2/4 GB LPDDR4, 16/32 GB eMMC | W-band DSP, PhotonNet inference, logging, dashboard and high-level IPC |
| Real-time MCU | STM32U585, Cortex-M33 up to 160 MHz, 786 KB SRAM, 2 MB Flash | deterministic acquisition, timestamps, trigger/control, watchdog, health |

The 4 GB/32 GB QRB2210 configuration is the preferred research target.

## Runtime boundary

```
W-band RF/IF -> ADC -> STM32U585 timestamp/trigger
                         |
                         v
                  IPC / shared transport
                         |
                         v
                 QRB2210 Linux
                   |        |
                   |        +--> logging/dashboard
                   v
             Radar DSP -> PhotonNet
                   |
                   v
              tracking/state
```

Do not allocate the V9 neural model to the STM32U585 by default. The STM32 side is a deterministic control/acquisition processor; the QRB2210 is the inference host.

## Benchmark rule

No latency number is inferred from clock frequency. Hardware claims require an on-device benchmark measuring:
- end-to-end T16 latency
- single-frame latency
- CPU utilization
- memory/RSS
- thermal behavior
- power
- IPC/data-copy overhead
- sustained 30 Hz operation
