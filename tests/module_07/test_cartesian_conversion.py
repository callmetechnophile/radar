"""Unit tests for Oxford polar to 2D Cartesian radar spatial grid conversion."""

import pytest
import numpy as np

from module_07_temporal.oxford_adapter import OxfordRadarAdapter


def test_cartesian_conversion_dimensions_and_finite_values():
    """Verify that Cartesian radar conversion produces valid 2D spatial grids without NaN/Inf."""
    adapter = OxfordRadarAdapter()
    frame = adapter.load_frame(0)

    cart_640 = adapter.get_cartesian_radar(frame, resolution_m_per_pixel=0.25, cart_size_pixels=640)
    assert cart_640.shape == (640, 640)
    assert cart_640.dtype == np.float32
    assert not np.isnan(cart_640).any()
    assert not np.isinf(cart_640).any()
    assert 0.0 <= np.min(cart_640) <= np.max(cart_640) <= 1.0


def test_cartesian_conversion_custom_resolutions():
    """Verify conversion at multiple grid resolutions (400x400 and 256x256)."""
    adapter = OxfordRadarAdapter()
    frame = adapter.load_frame(0)

    for sz in [256, 400]:
        cart = adapter.get_cartesian_radar(frame, resolution_m_per_pixel=0.20, cart_size_pixels=sz)
        assert cart.shape == (sz, sz)
        assert not np.isnan(cart).any()
