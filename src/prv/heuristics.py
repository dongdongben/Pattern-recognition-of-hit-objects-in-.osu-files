from __future__ import annotations

from typing import Dict, List


def apply_chunk_labels(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Build chunk labels from new-combo boundaries.
    - chunk_pos: on-screen number (1,2,3,...)
    - chunk_id: running chunk index per map
    """
    out: List[Dict[str, float]] = []
    chunk_id = 0
    chunk_pos = 0

    for i, row in enumerate(rows):
        row_copy = dict(row)
        is_new_combo = row_copy.get("is_new_combo", 0.0) >= 0.5

        if i == 0 or is_new_combo:
            chunk_id += 1
            chunk_pos = 1
        else:
            chunk_pos += 1

        row_copy["chunk_id"] = float(chunk_id)
        row_copy["chunk_pos"] = float(chunk_pos)

        out.append(row_copy)

    return out
