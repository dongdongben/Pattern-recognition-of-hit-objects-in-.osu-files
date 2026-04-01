from __future__ import annotations

from typing import Dict, List

from .osu_parser import ParsedBeatmap, get_active_beat_length_ms


def extract_object_features(parsed: ParsedBeatmap) -> List[Dict[str, float]]:
    """Return one feature row per hit object for chunk modeling."""
    rows: List[Dict[str, float]] = []
    objs = parsed.hit_objects

    for i, obj in enumerate(objs):
        prev_obj = objs[i - 1] if i > 0 else None

        dt_ms = float(obj.time_ms - prev_obj.time_ms) if prev_obj else 0.0
        beat_len = get_active_beat_length_ms(parsed.timing_points, obj.time_ms)
        dt_beats = dt_ms / beat_len if beat_len > 0 else 0.0

        rows.append(
            {
                "map_path": parsed.path,
                "object_index": float(i + 1),
                "time_ms": float(obj.time_ms),
                "x": float(obj.x),
                "y": float(obj.y),
                "dt_ms": dt_ms,
                "dt_beats": dt_beats,
                "is_slider": 1.0 if (obj.obj_type & 2) else 0.0,
                "is_spinner": 1.0 if (obj.obj_type & 8) else 0.0,
                "is_new_combo": 1.0 if obj.new_combo else 0.0,
                "chunk_id": float(obj.chunk_id),
                "chunk_pos": float(obj.chunk_pos),
            }
        )

    return rows
