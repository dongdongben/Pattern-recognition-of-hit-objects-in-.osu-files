from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import Dataset

FEATURE_COLUMNS = [
    "x",
    "y",
    "dt_ms",
    "dt_beats",
    "is_slider",
    "is_spinner",
]


class SequenceChunkDataset(Dataset):
    def __init__(self, csv_path: str, seq_len: int = 128):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.samples: List[Dict[str, torch.Tensor]] = []
        grouped_rows = self._load_grouped_rows()

        for rows in grouped_rows.values():
            for start in range(0, len(rows), seq_len):
                chunk = rows[start : start + seq_len]
                self.samples.append(self._build_sample(chunk))

    def _load_grouped_rows(self) -> Dict[str, List[Dict[str, str]]]:
        grouped_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)

        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue

                map_path = (row.get("map_path") or "").strip()
                object_index = (row.get("object_index") or "").strip()
                chunk_pos = (row.get("chunk_pos") or "").strip()
                if not map_path or not object_index or not chunk_pos:
                    continue

                row["map_path"] = map_path
                row["object_index"] = object_index
                row["chunk_pos"] = chunk_pos
                grouped_rows[map_path].append(row)

        for rows in grouped_rows.values():
            rows.sort(key=lambda row: int(float(row["object_index"])))

        return grouped_rows

    def _build_sample(self, rows: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        x = torch.zeros((self.seq_len, len(FEATURE_COLUMNS)), dtype=torch.float32)
        y = torch.full((self.seq_len,), -100, dtype=torch.long)
        mask = torch.zeros((self.seq_len,), dtype=torch.bool)

        for i, row in enumerate(rows):
            x[i] = torch.tensor([float(row[col]) for col in FEATURE_COLUMNS], dtype=torch.float32)
            chunk_pos = int(float(row["chunk_pos"]))
            y[i] = 1 if chunk_pos == 1 else 0
            mask[i] = True

        return {"x": x, "y": y, "mask": mask}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
