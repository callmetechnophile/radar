"""UNO Q hardware profiles for current PhotonNet/V9 edge deployment.

This module deliberately does not profile the legacy PhotonV0 model. V9/W-band
deployment is heterogeneous: QRB2210 runs the canonical neural model and heavy
DSP; STM32U585 handles deterministic real-time I/O and supervision.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from module_06_bitnet.runtime_specs import (
    HardwareProfile,
    UNO_Q_QRB2210_4GB,
    UNO_Q_QRB2210_2GB,
    UNO_Q_STM32U585,
)


V9_PARAMS = 24_777
V9_FP32_WEIGHT_BYTES = 96_790
V9_INT8_WEIGHT_BYTES = 24_200
V9_INT4_WEIGHT_BYTES = 12_100
V9_FP32_PEAK_RAM_BYTES = 140_660
V9_INT8_WEIGHT_PEAK_RAM_BYTES = 68_070
V9_T16_FLOPS = 16 * 50_774


def profile_v9(profile: HardwareProfile) -> Dict[str, Any]:
    """Report memory/compute fit without inventing MCU latency."""
    is_linux_host = "QRB2210" in profile.name
    weight_bytes = V9_INT8_WEIGHT_BYTES
    peak_ram = V9_INT8_WEIGHT_PEAK_RAM_BYTES

    return {
        "target": profile.name,
        "architecture": profile.architecture,
        "role": profile.role,
        "v9_parameters": V9_PARAMS,
        "v9_int8_weight_kb": weight_bytes / 1024.0,
        "v9_fp32_weight_kb": V9_FP32_WEIGHT_BYTES / 1024.0,
        "v9_int4_weight_kb": V9_INT4_WEIGHT_BYTES / 1024.0,
        "v9_int8_peak_ram_kb": peak_ram / 1024.0,
        "v9_fp32_peak_ram_kb": V9_FP32_PEAK_RAM_BYTES / 1024.0,
        "v9_t16_flops": V9_T16_FLOPS,
        "memory_fit": peak_ram < profile.sram_bytes,
        "deployment_class": "Linux application processor" if is_linux_host else "real-time MCU",
        "latency_status": (
            "Requires hardware benchmark on QRB2210; do not infer from clock rate."
            if is_linux_host
            else "Not the primary V9 inference target; use for control/IPC/timing."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile V9 against Arduino UNO Q targets.")
    parser.add_argument(
        "--target",
        choices=("qrb2210-4gb", "qrb2210-2gb", "stm32u585"),
        default="qrb2210-4gb",
    )
    args = parser.parse_args()

    profiles = {
        "qrb2210-4gb": UNO_Q_QRB2210_4GB,
        "qrb2210-2gb": UNO_Q_QRB2210_2GB,
        "stm32u585": UNO_Q_STM32U585,
    }
    result = profile_v9(profiles[args.target])

    print("=" * 72)
    print(f" PhotonShield AI — V9 Hardware Profile: {result['target']}")
    print("=" * 72)
    print(f"Role:                         {result['role']}")
    print(f"V9 parameters:                {result['v9_parameters']:,}")
    print(f"V9 INT8 weight size:         {result['v9_int8_weight_kb']:.2f} KB")
    print(f"V9 INT8 peak RAM:             {result['v9_int8_peak_ram_kb']:.2f} KB")
    print(f"V9 T16 compute estimate:      {result['v9_t16_flops']:,} FLOPs")
    print(f"Memory fit:                   {'PASS' if result['memory_fit'] else 'FAIL'}")
    print(f"Latency:                      {result['latency_status']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
