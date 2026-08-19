"""Unit tests for VoDObject3DHead."""

import pytest
import torch
from module_08_vod.object_head import VoDObject3DHead, NUM_VOD_CLASSES


def test_object_head_shapes():
    """Verify classification and 3D bounding box regression shapes."""
    head = VoDObject3DHead(in_dim=64, hidden_dim=64, num_classes=4)

    # Sequence [B, T, 64]
    z_seq = torch.randn(2, 8, 64)
    cls_logits, box_params = head(z_seq)

    assert cls_logits.shape == (2, 8, 4)
    assert box_params.shape == (2, 8, 7)
    assert not torch.isnan(cls_logits).any()
    assert not torch.isnan(box_params).any()

    # Single frame [B, 64]
    z_single = torch.randn(4, 64)
    cls_s, box_s = head(z_single)
    assert cls_s.shape == (4, 4)
    assert box_s.shape == (4, 7)
