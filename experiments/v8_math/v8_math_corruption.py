"""PhotonShield AI — V8-MATH Mathematical State Estimation Foundation
Module: v8_math_corruption.py

Deterministic temporal corruption generator matching V7.9 protocols exactly.
Generates observation masks m_t in {0, 1} for Bernoulli dropout and contiguous gaps.
"""

from typing import Tuple, List, Optional
import numpy as np


class TemporalCorruptionGenerator:
    """Generates deterministic temporal missingness masks."""
    def __init__(self, default_seed: int = 777):
        self.default_seed = default_seed

    def generate_bernoulli_mask(self, T: int, rate: float, seed: Optional[int] = None) -> np.ndarray:
        """Generates Bernoulli dropout mask [T]: 1 = observed, 0 = missing."""
        if rate <= 0.0:
            return np.ones(T, dtype=np.int32)
        rng = np.random.default_rng(seed if seed is not None else self.default_seed)
        mask = (rng.random(T) > rate).astype(np.int32)
        # Ensure at least frame 0 is observed to initialize estimator
        mask[0] = 1
        return mask

    def generate_contiguous_gap_mask(self, T: int, gap_length: int, start_idx: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
        """Generates contiguous gap mask [T]: 1 = observed, 0 = missing."""
        mask = np.ones(T, dtype=np.int32)
        if gap_length <= 0:
            return mask
        if start_idx is None:
            rng = np.random.default_rng(seed if seed is not None else self.default_seed)
            max_start = max(1, T - gap_length)
            start_idx = int(rng.integers(1, max_start))
        end_idx = min(T, start_idx + gap_length)
        mask[start_idx:end_idx] = 0
        return mask

    def generate_contiguous_dropout_mask(self, T: int, rate: float, seed: Optional[int] = None) -> np.ndarray:
        """Contiguous dropout where gap length = round(T * rate)."""
        gap = int(round(T * rate))
        return self.generate_contiguous_gap_mask(T, gap, seed=seed)
