from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

FEATURE_COLUMNS = [
    "dt_ms",
    "dt_beats",
    "dx",
    "dy",
    "distance",
    "norm_distance",
    "turn_angle_deg",
    "is_slider",
    "is_spinner",
]


class SequenceChunkDataset(Dataset):
    def __init__(self, csv_path: str, seq_len: int = 128):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.samples: List[Dict[str, torch.Tensor]] = []
        raise NotImplementedError("TODO: load csv and build fixed-length sequence chunks")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
