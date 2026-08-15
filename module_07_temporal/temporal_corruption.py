"""Deterministic temporal corruption utility for Oxford Radar sequences."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class TemporalRadarCorruption:
    """Applies reproducible temporal frame dropouts and contiguous missing gaps to radar sequences."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def apply_random_dropout(
        self,
        sequence_length: int,
        p_drop: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply Bernoulli temporal frame dropout.

        Args:
            sequence_length: Number of frames in sequence T.
            p_drop: Probability of dropping a frame (0.0 to 1.0).

        Returns:
            Tuple of (binary mask of shape [T] where 1=observed, 0=dropped, corruption stats dict).
        """
        if p_drop <= 0.0:
            mask = np.ones(sequence_length, dtype=np.float32)
        elif p_drop >= 1.0:
            mask = np.zeros(sequence_length, dtype=np.float32)
            mask[0] = 1.0  # Keep at least initial frame if clamped
        else:
            mask = (self.rng.rand(sequence_length) >= p_drop).astype(np.float32)

        stats = self._analyze_mask(mask)
        stats["p_drop_target"] = p_drop
        return mask, stats

    def apply_contiguous_gap(
        self,
        sequence_length: int,
        gap_length: int,
        start_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply a contiguous block gap of missing frames.

        Args:
            sequence_length: Number of frames in sequence T.
            gap_length: Number of consecutive frames to drop.
            start_idx: Starting index of the gap (random if None).

        Returns:
            Tuple of (binary mask of shape [T], corruption stats dict).
        """
        mask = np.ones(sequence_length, dtype=np.float32)
        gap_len = min(gap_length, sequence_length)

        if start_idx is None:
            max_start = max(0, sequence_length - gap_len)
            start_idx = int(self.rng.randint(0, max_start + 1)) if max_start > 0 else 0

        end_idx = min(sequence_length, start_idx + gap_len)
        mask[start_idx:end_idx] = 0.0

        stats = self._analyze_mask(mask)
        stats["gap_length_target"] = gap_length
        stats["gap_start_idx"] = start_idx
        stats["gap_end_idx"] = end_idx
        return mask, stats

    def _analyze_mask(self, mask: np.ndarray) -> Dict[str, Any]:
        """Compute gap lengths, counts, and ratios from a binary observation mask."""
        T = len(mask)
        dropped_count = int(np.sum(mask == 0.0))
        drop_ratio = dropped_count / float(T) if T > 0 else 0.0

        # Find contiguous gaps (runs of zeros)
        gap_lengths = []
        current_gap = 0
        for val in mask:
            if val == 0.0:
                current_gap += 1
            else:
                if current_gap > 0:
                    gap_lengths.append(current_gap)
                    current_gap = 0
        if current_gap > 0:
            gap_lengths.append(current_gap)

        num_gaps = len(gap_lengths)
        mean_gap_len = float(np.mean(gap_lengths)) if num_gaps > 0 else 0.0
        max_gap_len = int(np.max(gap_lengths)) if num_gaps > 0 else 0
        gaps_ge_3_pct = (
            (sum(1 for g in gap_lengths if g >= 3) / num_gaps) * 100.0 if num_gaps > 0 else 0.0
        )

        return {
            "sequence_length": T,
            "missing_frame_count": dropped_count,
            "missing_frame_ratio": drop_ratio,
            "number_of_gaps": num_gaps,
            "mean_gap_length": mean_gap_len,
            "max_gap_length": max_gap_len,
            "percentage_gaps_ge_3": gaps_ge_3_pct,
            "gap_lengths": gap_lengths,
        }
