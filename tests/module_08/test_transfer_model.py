"""Unit tests for VoDTransfer3DModel across all 6 regimes."""

import pytest
import torch
from module_08_vod.transfer_model import VoDTransfer3DModel


@pytest.mark.parametrize("regime", [
    "native_no_physics",
    "frozen_transfer",
    "physics_transfer",
    "partial_finetune",
    "full_finetune",
    "native_with_physics",
])
def test_transfer_model_regimes_and_gradients(regime):
    """Verify forward and backward execution under each regime."""
    model = VoDTransfer3DModel(regime=regime, point_in_dim=7, feature_dim=64, hidden_dim=64)
    tokens = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8, 1)

    cls_logits, box_params, occ_logits, kin = model(tokens, mask)

    assert cls_logits.shape == (2, 8, 4)
    assert box_params.shape == (2, 8, 7)
    assert occ_logits.shape == (2, 8, 32, 32, 8)
    assert kin.shape == (2, 8, 5)

    loss = cls_logits.sum() + box_params.sum() + occ_logits.sum() + kin.sum()
    loss.backward()

    if regime in ("frozen_transfer", "physics_transfer"):
        # Mamba backbone should have NO gradients
        for p in model.mamba_layers.parameters():
            assert p.grad is None
        # Object head MUST have gradients
        for p in model.object_head.parameters():
            assert p.grad is not None
