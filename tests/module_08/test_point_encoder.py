"""Unit tests for RadarPointEncoder."""

import torch
import pytest
from module_08_vod.radar_point_encoder import RadarPointEncoder


def test_point_encoder_shapes():
    """Verify point encoder forward pass with batched and unbatched inputs."""
    encoder = RadarPointEncoder(in_channels=7, hidden_dim=32, out_dim=64, pooling="max")

    # 1. Unbatched [N, 7]
    pts = torch.randn(150, 7)
    token = encoder(pts)
    assert token.shape == (64,)

    # 2. Batched [B, N, 7]
    batch_pts = torch.randn(4, 200, 7)
    batch_tokens = encoder(batch_pts)
    assert batch_tokens.shape == (4, 64)


def test_point_encoder_permutation_invariance():
    """Verify that permutation of point order does NOT change the output frame token."""
    encoder = RadarPointEncoder(in_channels=7, hidden_dim=32, out_dim=64, pooling="max")
    encoder.eval()

    torch.manual_seed(42)
    pts = torch.randn(100, 7)
    perm = torch.randperm(100)
    pts_perm = pts[perm]

    with torch.no_grad():
        token1 = encoder(pts)
        token2 = encoder(pts_perm)

    diff = torch.max(torch.abs(token1 - token2)).item()
    assert diff < 1e-5, f"Permutation invariance failed: max diff = {diff}"


def test_point_encoder_empty_point_cloud():
    """Verify graceful handling of empty point cloud."""
    encoder = RadarPointEncoder(in_channels=7, hidden_dim=32, out_dim=64, pooling="max")
    pts_empty = torch.zeros(0, 7)
    token = encoder(pts_empty)
    assert token.shape == (64,)
    assert torch.all(token == 0.0)
