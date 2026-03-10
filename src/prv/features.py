from __future__ import annotations

import math
from typing import Dict, List

from .osu_parser import ParsedBeatmap, get_active_beat_length_ms

CONTEXT_WINDOW = 6


def _angle_between(v1x: float, v1y: float, v2x: float, v2y: float) -> float:
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    dot = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(dot))


def extract_object_features(parsed: ParsedBeatmap, circle_radius_px: float = 64.0) -> List[Dict[str, float]]:
    """Return one feature row per hit object with chunk labels and 6-note context."""
    rows: List[Dict[str, float]] = []
    objs = parsed.hit_objects

    for i, obj in enumerate(objs):
        prev_obj = objs[i - 1] if i > 0 else None
        prev2_obj = objs[i - 2] if i > 1 else None

        dt_ms = float(obj.time_ms - prev_obj.time_ms) if prev_obj else 0.0
        beat_len = get_active_beat_length_ms(parsed.timing_points, obj.time_ms)
        dt_beats = dt_ms / beat_len if beat_len > 0 else 0.0

        dx = float(obj.x - prev_obj.x) if prev_obj else 0.0
        dy = float(obj.y - prev_obj.y) if prev_obj else 0.0
        dist = math.hypot(dx, dy)
        norm_dist = dist / circle_radius_px if circle_radius_px > 0 else 0.0

        prev_dx = float(prev_obj.x - prev2_obj.x) if prev_obj and prev2_obj else 0.0
        prev_dy = float(prev_obj.y - prev2_obj.y) if prev_obj and prev2_obj else 0.0
        turn_angle_deg = 180.0 - _angle_between(prev_dx, prev_dy, dx, dy)

        row: Dict[str, float] = {
            "map_path": parsed.path,
            "object_index": float(i + 1),
            "time_ms": float(obj.time_ms),
            "x": float(obj.x),
            "y": float(obj.y),
            "dt_ms": dt_ms,
            "dt_beats": dt_beats,
            "dx": dx,
            "dy": dy,
            "distance": dist,
            "norm_distance": norm_dist,
            "turn_angle_deg": turn_angle_deg,
            "is_slider": 1.0 if (obj.obj_type & 2) else 0.0,
            "is_spinner": 1.0 if (obj.obj_type & 8) else 0.0,
            "is_new_combo": 1.0 if obj.new_combo else 0.0,
            "chunk_id": float(obj.chunk_id),
            "chunk_pos": float(obj.chunk_pos),
        }

        # 6-note history window ending at current object (i-5 ... i).
        # slot_0 is oldest in window, slot_5 is current object.
        for slot in range(CONTEXT_WINDOW):
            j = i - (CONTEXT_WINDOW - 1 - slot)
            if j < 0:
                row[f"ctx_valid_{slot}"] = 0.0
                row[f"ctx_rel_x_{slot}"] = 0.0
                row[f"ctx_rel_y_{slot}"] = 0.0
                row[f"ctx_rel_dt_beats_{slot}"] = 0.0
                continue

            ctx_obj = objs[j]
            row[f"ctx_valid_{slot}"] = 1.0
            row[f"ctx_rel_x_{slot}"] = float(ctx_obj.x - obj.x)
            row[f"ctx_rel_y_{slot}"] = float(ctx_obj.y - obj.y)
            row[f"ctx_rel_dt_beats_{slot}"] = float(ctx_obj.time_ms - obj.time_ms) / beat_len if beat_len > 0 else 0.0

        rows.append(row)

    return rows
