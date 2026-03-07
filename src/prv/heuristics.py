from __future__ import annotations

from typing import Dict, List


def label_level1(row: Dict[str, float]) -> str:
    """Assign heuristic Level-1 label."""
    raise NotImplementedError("TODO: implement Level-1 heuristic")


def label_level2(row: Dict[str, float], prev_row: Dict[str, float] | None, level1: str) -> str:
    """Assign heuristic Level-2 label."""
    raise NotImplementedError("TODO: implement Level-2 heuristic")


def apply_heuristic_labels(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Append label_level1 and label_level2 to each feature row."""
    raise NotImplementedError("TODO: implement pseudo-labeling pass")
