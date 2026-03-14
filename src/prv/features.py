from __future__ import annotations

import math
from typing import Dict, List

from .osu_parser import ParsedBeatmap, get_active_beat_length_ms


def extract_object_features(parsed: ParsedBeatmap, circle_radius_px: float = 64.0) -> List[Dict[str, float]]:
    """Return one feature row per hit object for chunk modeling."""
    rows: List[Dict[str, float]] = []
    objs = parsed.hit_objects

    for i, obj in enumerate(objs):
        prev_obj = objs[i - 1] if i > 0 else None

        dt_ms = float(obj.time_ms - prev_obj.time_ms) if prev_obj else 0.0
        beat_len = get_active_beat_length_ms(parsed.timing_points, obj.time_ms)
        dt_beats = dt_ms / beat_len if beat_len > 0 else 0.0

        dx = float(obj.x - prev_obj.x) if prev_obj else 0.0
        dy = float(obj.y - prev_obj.y) if prev_obj else 0.0
        dist = math.hypot(dx, dy)
        norm_dist = dist / circle_radius_px if circle_radius_px > 0 else 0.0

        rows.append(
            {
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
                "is_slider": 1.0 if (obj.obj_type & 2) else 0.0,
                "is_spinner": 1.0 if (obj.obj_type & 8) else 0.0,
                "is_new_combo": 1.0 if obj.new_combo else 0.0,
                "chunk_id": float(obj.chunk_id),
                "chunk_pos": float(obj.chunk_pos),
            }
        )

    return rows
