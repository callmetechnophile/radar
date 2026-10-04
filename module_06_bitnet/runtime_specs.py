"""Hardware runtime specifications for the heterogeneous Arduino UNO Q platform.

The UNO Q is a heterogeneous board:
- QRB2210 Linux side: application-class CPU/GPU with 2 GB or 4 GB LPDDR4 and
  16 GB or 32 GB eMMC.
- STM32U585 MCU side: deterministic real-time controller with 786 KB SRAM and
  2 MB Flash.

The PhotonNet/W-band edge architecture uses the QRB2210 for DSP + neural
inference and the STM32U585 for deterministic I/O, timing, watchdog, and
low-level control. These are separate runtime profiles; they must not be
treated as one MCU memory budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware profile specification."""

    name: str
    architecture: str
    flash_bytes: int
    sram_bytes: int
    storage_bytes: int
    clock_frequency_hz: int
    max_macs_budget: int
    fpu_support: bool
    os: str
    role: str


UNO_Q_QRB2210_4GB = HardwareProfile(
    name="Arduino UNO Q — QRB2210 4GB",
    architecture="4x Cortex-A53 + Adreno 702",
    flash_bytes=0,
    sram_bytes=4 * 1024**3,
    storage_bytes=32 * 1024**3,
    clock_frequency_hz=2_000_000_000,
    max_macs_budget=0,
    fpu_support=True,
    os="Debian Linux",
    role="W-band DSP, PhotonNet inference, data processing, dashboard/IPC host",
)

UNO_Q_QRB2210_2GB = HardwareProfile(
    name="Arduino UNO Q — QRB2210 2GB",
    architecture="4x Cortex-A53 + Adreno 702",
    flash_bytes=0,
    sram_bytes=2 * 1024**3,
    storage_bytes=16 * 1024**3,
    clock_frequency_hz=2_000_000_000,
    max_macs_budget=0,
    fpu_support=True,
    os="Debian Linux",
    role="W-band DSP, PhotonNet inference, data processing, dashboard/IPC host",
)

UNO_Q_STM32U585 = HardwareProfile(
    name="Arduino UNO Q — STM32U585",
    architecture="Cortex-M33",
    flash_bytes=2 * 1024**2,
    sram_bytes=786 * 1024,
    storage_bytes=0,
    clock_frequency_hz=160_000_000,
    max_macs_budget=1_000_000,
    fpu_support=True,
    os="Bare metal / RTOS",
    role="Deterministic acquisition, timing, watchdog, trigger, low-level control",
)

# Backward-compatible alias. New code should select an explicit profile.
ARDUINO_UNO_Q_PROFILE = UNO_Q_QRB2210_4GB

STM32_H7_PROFILE = HardwareProfile(
    name="STM32H7 High-Performance Edge MCU",
    architecture="Cortex-M7",
    flash_bytes=2 * 1024**2,
    sram_bytes=1024 * 1024,
    storage_bytes=0,
    clock_frequency_hz=480_000_000,
    max_macs_budget=2_000_000,
    fpu_support=True,
    os="Bare metal / RTOS",
    role="Generic high-performance MCU reference",
)

GENERIC_CORTEX_M4_PROFILE = HardwareProfile(
    name="Generic ARM Cortex-M4 Microcontroller",
    architecture="Cortex-M4",
    flash_bytes=256 * 1024,
    sram_bytes=32 * 1024,
    storage_bytes=0,
    clock_frequency_hz=48_000_000,
    max_macs_budget=80_000,
    fpu_support=False,
    os="Bare metal / RTOS",
    role="Generic legacy MCU reference",
)


def profile_summary(profile: HardwareProfile) -> Dict[str, Any]:
    """Return a normalized, serializable profile summary."""
    return {
        "name": profile.name,
        "architecture": profile.architecture,
        "flash_mb": profile.flash_bytes / 1024**2,
        "sram_mb": profile.sram_bytes / 1024**2,
        "storage_gb": profile.storage_bytes / 1024**3,
        "clock_ghz": profile.clock_frequency_hz / 1e9,
        "max_macs_budget": profile.max_macs_budget,
        "fpu_support": profile.fpu_support,
        "os": profile.os,
        "role": profile.role,
    }
