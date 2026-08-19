"""Unit tests for AnchorBasedMultiObjectHead and QueryBasedMultiObjectHead."""

import pytest
import torch
from module_08_vod.multi_object_head import AnchorBasedMultiObjectHead, QueryBasedMultiObjectHead


def test_anchor_based_head():
    """Verify anchor-based multi-object prediction shapes and gradient flow."""
    head = AnchorBasedMultiObjectHead(in_dim=64, hidden_dim=64, num_anchors=16, num_classes=3)
    z = torch.randn(2, 8, 64, requires_grad=True)

    confs, classes, boxes = head(z)
    assert confs.shape == (2, 8, 16, 1)
    assert classes.shape == (2, 8, 16, 3)
    assert boxes.shape == (2, 8, 16, 7)

    loss = confs.sum() + classes.sum() + boxes.sum()
    loss.backward()
    assert z.grad is not None


def test_query_based_head():
    """Verify query-based multi-object prediction shapes and gradient flow."""
    head = QueryBasedMultiObjectHead(in_dim=64, hidden_dim=64, num_queries=16, num_classes=3)
    z = torch.randn(2, 8, 64, requires_grad=True)

    confs, classes, boxes = head(z)
    assert confs.shape == (2, 8, 16, 1)
    assert classes.shape == (2, 8, 16, 3)
    assert boxes.shape == (2, 8, 16, 7)

    loss = confs.sum() + classes.sum() + boxes.sum()
    loss.backward()
    assert z.grad is not None
