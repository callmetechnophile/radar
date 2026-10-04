"""Hardware backend capability abstraction for current edge targets."""

from __future__ import annotations
from typing import Dict, Any
import torch

HARDWARE_DISCLAIMER = (
    "Low-bit model results are model-level results. Native QRB2210 or STM32U585 "
    "acceleration has not been benchmarked until an on-device test is recorded."
)


def get_hardware_backend_info(backend_name: str = "pytorch") -> Dict[str, Any]:
    """Return runtime information without implying native ternary acceleration."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "backend": backend_name,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "uno_q_inference_host": "QRB2210",
        "uno_q_control_host": "STM32U585",
        "supports_native_ternary_hardware_ops": False,
        "disclaimer": HARDWARE_DISCLAIMER,
    }
