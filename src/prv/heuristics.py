from __future__ import annotations

from typing import Dict, List


def label_level1(row: Dict[str, float]) -> str:
    if row["is_slider"] >= 0.5:
        return "sliders"
    if row["dt_beats"] <= 0.35 and row["norm_distance"] <= 1.8:
        return "streams"
    if row["norm_distance"] >= 2.4:
        return "jumps"
    return "filler"


def label_level2(row: Dict[str, float], prev_row: Dict[str, float] | None, level1: str) -> str:
    if level1 == "sliders":
        return "slider_aim"

    if level1 == "jumps":
        angle = row["turn_angle_deg"]
        if angle <= 40.0 or angle >= 130.0:
            return "awkward_aim"
        return "jump_aim"

    if level1 == "streams":
        if prev_row is not None:
            prev_dt = prev_row["dt_beats"]
            cur_dt = row["dt_beats"]
            if prev_dt - cur_dt > 0.12:
                return "accelerating_flow_aim"
        return "flow_aim"

    return "linear_aim"


def apply_heuristic_labels(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    prev = None
    for row in rows:
        row_copy = dict(row)
        l1 = label_level1(row_copy)
        l2 = label_level2(row_copy, prev, l1)
        row_copy["label_level1"] = l1
        row_copy["label_level2"] = l2
        out.append(row_copy)
        prev = row_copy
    return out
