"""Unit tests for sequence builder and continuous snippet partitioning."""

import pytest
from pathlib import Path
from module_08_vod.constants import IMAGESETS_DIR
from module_08_vod.sequence_builder import (
    extract_continuous_snippets,
    build_100_sequence_split,
    compute_training_normalization,
)


def test_extract_continuous_snippets():
    """Verify continuous snippet identification from frame index sequences."""
    test_ids = [0, 1, 2, 3, 10, 11, 12, 50, 51, 52, 53, 54]
    snippets = extract_continuous_snippets(test_ids)
    assert len(snippets) == 3
    assert snippets[0] == [0, 1, 2, 3]
    assert snippets[1] == [10, 11, 12]
    assert snippets[2] == [50, 51, 52, 53, 54]


def test_build_100_sequence_split():
    """Verify 100-sequence partition without sequence boundary crossing."""
    train_txt = IMAGESETS_DIR / "train.txt"
    if not train_txt.exists():
        pytest.skip("VoD ImageSets not available.")

    split = build_100_sequence_split(train_txt, seq_len=8, num_train=70, num_val=15, num_test=15)
    assert split["num_train"] == 70
    assert split["num_val"] == 15
    assert split["num_test"] == 15
    assert len(split["train"]) == 70
    assert len(split["val"]) == 15
    assert len(split["test"]) == 15

    # Check each sequence is exactly length 8 and consecutive
    for seq in split["train"] + split["val"] + split["test"]:
        assert len(seq) == 8
        for i in range(len(seq) - 1):
            assert seq[i + 1] == seq[i] + 1
