from __future__ import annotations

import math
from typing import Dict, List

from .osu_parser import ParsedBeatmap, get_active_beat_length_ms


def _angle_between(v1x: float, v1y: float, v2x: float, v2y: float) -> float:
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    dot = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(dot))


def extract_object_features(parsed: ParsedBeatmap, circle_radius_px: float = 64.0) -> List[Dict[str, float]]:
    objs = parsed.hit_objects
    rows: List[Dict[str, float]] = []

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
        turn_angle_deg = _angle_between(prev_dx, prev_dy, dx, dy)

        rows.append(
            {
                "map_path": parsed.path,
                "object_index": float(i),
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
                "is_slider": 1.0 if obj.is_slider else 0.0,
                "is_spinner": 1.0 if obj.is_spinner else 0.0,
            }
        )
    return rows
