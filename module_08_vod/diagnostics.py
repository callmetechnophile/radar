"""Physical Diagnostics and Edge Deployment Footprint Audit for VoD Phase V6.1."""

from __future__ import annotations

import time
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn


def check_physical_plausibility(
    radar_pts: np.ndarray,
    pred_occ: np.ndarray,
) -> Dict[str, float]:
    """Calculate diagnostic physical checks on inputs and reconstructed 3D representation."""
    # Radial Doppler velocity range
    v_r = radar_pts[:, 4]
    rcs = radar_pts[:, 3]

    active_voxels = (pred_occ >= 0.5).sum()
    total_voxels = pred_occ.size

    return {
        "mean_radar_velocity_mps": float(np.mean(v_r)),
        "std_radar_velocity_mps": float(np.std(v_r)),
        "mean_radar_rcs_dbsm": float(np.mean(rcs)),
        "active_occupancy_ratio": float(active_voxels / total_voxels),
        "point_density_ratio": float(len(radar_pts) / max(1, active_voxels)),
    }


def audit_model_edge_footprint(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 8, 64),
    device: str = "cpu",
    warmup_iters: int = 10,
    measure_iters: int = 50,
) -> Dict[str, float]:
    """Benchmark model parameter count, memory footprint, and inference latency."""
    model.eval()
    model.to(device)

    # 1. Parameter count and weight memory
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weight_mem_mb = (total_params * 4) / (1024 * 1024)  # FP32: 4 bytes per param

    # 2. Synthetic input tensor
    dummy_input = torch.randn(*input_shape, device=device)
    dummy_mask = torch.ones(input_shape[0], input_shape[1], 1, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input, dummy_mask)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Measure latency
    latencies_ms = []
    with torch.no_grad():
        for _ in range(measure_iters):
            t0 = time.perf_counter()
            _ = model(dummy_input, dummy_mask)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    # Estimate FLOPs (approximate MAC count for linear layers)
    # Forward pass FLOPs estimate
    flops_est = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops_est += 2 * m.in_features * m.out_features

    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "weight_memory_mb": float(weight_mem_mb),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "median_latency_ms": float(np.median(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "approx_mflop_per_pass": float(flops_est / 1e6),
    }
