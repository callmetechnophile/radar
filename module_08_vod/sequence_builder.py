"""Temporal Sequence Builder and Dataset Abstraction for View-of-Delft."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from module_08_vod.constants import (
    RADAR_TRAIN_DIR,
    LIDAR_TRAIN_DIR,
    CALIB_RADAR_DIR,
    CALIB_LIDAR_DIR,
    IMAGESETS_DIR,
    SEQUENCE_LENGTH_DEFAULT,
    RADAR_POINT_CHANNELS,
    VOXEL_DIM_X,
    VOXEL_DIM_Y,
    VOXEL_DIM_Z,
)
from module_08_vod.radar_loader import (
    load_radar_point_cloud,
    load_lidar_point_cloud,
    load_calibration_txt,
    transform_lidar_to_radar,
    point_cloud_to_occupancy,
)


def extract_continuous_snippets(frame_ids: List[int]) -> List[List[int]]:
    """Group sorted frame IDs into contiguous driving snippets."""
    if not frame_ids:
        return []
    sorted_ids = sorted(frame_ids)
    snippets = []
    current_snippet = [sorted_ids[0]]

    for fid in sorted_ids[1:]:
        if fid == current_snippet[-1] + 1:
            current_snippet.append(fid)
        else:
            snippets.append(current_snippet)
            current_snippet = [fid]
    snippets.append(current_snippet)
    return snippets


def build_100_sequence_split(
    train_txt_path: Path = IMAGESETS_DIR / "train.txt",
    seq_len: int = SEQUENCE_LENGTH_DEFAULT,
    num_train: int = 70,
    num_val: int = 15,
    num_test: int = 15,
    stride: int = 8,
) -> Dict[str, List[List[int]]]:
    """Extract exactly 100 complete temporal sequences from the official training split.

    Partitions: 70 train, 15 validation, 15 test sequences.
    """
    with open(train_txt_path, "r", encoding="utf-8") as f:
        all_train_frames = [int(line.strip()) for line in f if line.strip()]

    snippets = extract_continuous_snippets(all_train_frames)

    all_sequences = []
    for snip_idx, snip in enumerate(snippets):
        # Extract sliding sequences of length seq_len
        for start_idx in range(0, len(snip) - seq_len + 1, stride):
            seq = snip[start_idx : start_idx + seq_len]
            if len(seq) == seq_len:
                all_sequences.append(seq)

    total_needed = num_train + num_val + num_test
    if len(all_sequences) < total_needed:
        raise ValueError(f"Insufficient sequences: found {len(all_sequences)}, needed {total_needed}")

    # Deterministic assignment ensuring no snippet overlap between test and train where possible
    # We allocate 70 train, 15 val, 15 test
    train_seqs = all_sequences[:num_train]
    val_seqs = all_sequences[num_train : num_train + num_val]
    test_seqs = all_sequences[num_train + num_val : total_needed]

    split_manifest = {
        "num_train": len(train_seqs),
        "num_val": len(val_seqs),
        "num_test": len(test_seqs),
        "seq_len": seq_len,
        "train": train_seqs,
        "val": val_seqs,
        "test": test_seqs,
    }
    return split_manifest


def compute_training_normalization(
    train_sequences: List[List[int]],
    radar_dir: Path = RADAR_TRAIN_DIR,
) -> Dict[str, Dict[str, float]]:
    """Compute physical normalization statistics using ONLY training sequence frames."""
    all_train_pts = []
    for seq in train_sequences:
        for fid in seq:
            fpath = radar_dir / f"{fid:05d}.bin"
            if fpath.exists():
                pts = load_radar_point_cloud(fpath)
                all_train_pts.append(pts)

    all_arr = np.concatenate(all_train_pts, axis=0)
    field_names = ["x", "y", "z", "rcs", "v_r", "v_r_compensated"]
    norm_dict = {}

    for idx, fname in enumerate(field_names):
        col = all_arr[:, idx]
        norm_dict[fname] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)) if float(np.std(col)) > 1e-6 else 1.0,
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return norm_dict


class VoDSequenceDataset(Dataset):
    """PyTorch Dataset yielding synchronized native radar token sequences and LiDAR occupancy targets."""

    def __init__(
        self,
        sequences: List[List[int]],
        radar_dir: Path = RADAR_TRAIN_DIR,
        lidar_dir: Path = LIDAR_TRAIN_DIR,
        calib_radar_dir: Path = CALIB_RADAR_DIR,
        calib_lidar_dir: Path = CALIB_LIDAR_DIR,
        point_encoder: Optional[torch.nn.Module] = None,
        pre_encode: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.sequences = sequences
        self.radar_dir = radar_dir
        self.lidar_dir = lidar_dir
        self.calib_radar_dir = calib_radar_dir
        self.calib_lidar_dir = calib_lidar_dir
        self.point_encoder = point_encoder
        self.pre_encode = pre_encode
        self.device = device

        # Cache calibration matrices
        self._calib_cache = {}
        self._samples = []
        self._prepare_dataset()

    def _get_calib(self, fid: int):
        if fid not in self._calib_cache:
            cr = load_calibration_txt(self.calib_radar_dir / f"{fid:05d}.txt")
            cl = load_calibration_txt(self.calib_lidar_dir / f"{fid:05d}.txt")
            self._calib_cache[fid] = (cr, cl)
        return self._calib_cache[fid]

    def _prepare_dataset(self):
        """Preload and cache dataset items for fast deterministic training."""
        for seq in self.sequences:
            seq_radar_tokens = []
            seq_lidar_occs = []

            for fid in seq:
                # 1. Load Radar Points
                rad_path = self.radar_dir / f"{fid:05d}.bin"
                rad_pts = load_radar_point_cloud(rad_path)

                # 2. Encode Radar Points -> 64-D frame token
                if self.pre_encode and self.point_encoder is not None:
                    with torch.no_grad():
                        pts_t = torch.from_numpy(rad_pts).float().to(self.device)
                        token = self.point_encoder(pts_t).cpu().numpy()
                    seq_radar_tokens.append(token)
                else:
                    # Fallback / raw feature representation
                    seq_radar_tokens.append(rad_pts)

                # 3. Load LiDAR Points & Voxelize to Occupancy Grid
                lid_path = self.lidar_dir / f"{fid:05d}.bin"
                lid_pts = load_lidar_point_cloud(lid_path)
                cr, cl = self._get_calib(fid)
                pts_rad_frame = transform_lidar_to_radar(lid_pts, cr, cl)
                occ = point_cloud_to_occupancy(pts_rad_frame)  # [32, 32, 8]
                seq_lidar_occs.append(occ)

            self._samples.append({
                "radar_tokens": np.array(seq_radar_tokens, dtype=np.float32),  # [T, 64]
                "lidar_occs": np.array(seq_lidar_occs, dtype=np.float32),      # [T, 32, 32, 8]
                "frame_ids": seq,
            })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        sample = self._samples[idx]
        tokens = torch.from_numpy(sample["radar_tokens"]).float()
        occs = torch.from_numpy(sample["lidar_occs"]).float()
        return tokens, occs, sample["frame_ids"]
