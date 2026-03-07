from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .labels import LEVEL1_TO_ID, LEVEL2_TO_ID

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
        self.seq_len = seq_len
        self.samples: List[Dict[str, torch.Tensor]] = []
        grouped = self._read_grouped(csv_path)

        for _, rows in grouped.items():
            for start in range(0, len(rows), seq_len):
                chunk = rows[start : start + seq_len]
                self.samples.append(self._tensorize_chunk(chunk))

    def _read_grouped(self, csv_path: str) -> Dict[str, List[Dict[str, str]]]:
        grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                grouped[row["map_path"]].append(row)

        for key in grouped:
            grouped[key].sort(key=lambda r: int(float(r["object_index"])))
        return grouped

    def _tensorize_chunk(self, rows: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        x = torch.zeros((self.seq_len, len(FEATURE_COLUMNS)), dtype=torch.float32)
        y1 = torch.full((self.seq_len,), -100, dtype=torch.long)
        y2 = torch.full((self.seq_len,), -100, dtype=torch.long)
        mask = torch.zeros((self.seq_len,), dtype=torch.bool)

        for i, row in enumerate(rows):
            x[i] = torch.tensor([float(row[c]) for c in FEATURE_COLUMNS], dtype=torch.float32)
            y1[i] = LEVEL1_TO_ID[row["label_level1"]]
            y2[i] = LEVEL2_TO_ID[row["label_level2"]]
            mask[i] = True

        return {"x": x, "y1": y1, "y2": y2, "mask": mask}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
