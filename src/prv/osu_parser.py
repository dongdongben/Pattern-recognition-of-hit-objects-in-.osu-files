from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TimingPoint:
    time_ms: float
    beat_length_ms: float
    meter: int
    uninherited: bool


@dataclass
class HitObject:
    x: int
    y: int
    time_ms: int
    obj_type: int
    raw_params: str
    new_combo: bool
    chunk_id: int = 0
    chunk_pos: int = 0


@dataclass
class ParsedBeatmap:
    path: str
    timing_points: List[TimingPoint]
    hit_objects: List[HitObject]


def parse_osu_file(path: str | Path) -> ParsedBeatmap:
    """Parse [TimingPoints] and [HitObjects] from a .osu file."""
    timing_points: List[TimingPoint] = []
    hit_objects: List[HitObject] = []
    current_section = None
    path = Path(path)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line
                continue

            if current_section == "[TimingPoints]":
                parts = line.split(",")
                if len(parts) >= 7:
                    try:
                        timing_points.append(
                            TimingPoint(
                                time_ms=float(parts[0]),
                                beat_length_ms=float(parts[1]),
                                meter=int(parts[2]),
                                uninherited=(parts[6] == "1"),
                            )
                        )
                    except ValueError:
                        pass

            elif current_section == "[HitObjects]":
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    try:
                        obj_type = int(parts[3])
                        raw_params = ",".join(parts[5:]) if len(parts) > 5 else ""
                        hit_objects.append(
                            HitObject(
                                x=int(parts[0]),
                                y=int(parts[1]),
                                time_ms=int(parts[2]),
                                obj_type=obj_type,
                                raw_params=raw_params,
                                new_combo=bool(obj_type & 4),
                            )
                        )
                    except ValueError:
                        pass

    # Derive chunk labels from combo boundaries (new combo resets on-screen numbering).
    chunk_id = 0
    chunk_pos = 0
    for i, obj in enumerate(hit_objects):
        if i == 0 or obj.new_combo:
            chunk_id += 1
            chunk_pos = 1
        else:
            chunk_pos += 1

        obj.chunk_id = chunk_id
        obj.chunk_pos = chunk_pos

    return ParsedBeatmap(
        path=str(path),
        timing_points=timing_points,
        hit_objects=hit_objects,
    )


def get_active_beat_length_ms(timing_points: List[TimingPoint], time_ms: int) -> float:
    """Return the active uninherited beat length at a specific timestamp."""
    if not timing_points:
        return 500.0

    current_beat_length_ms = 500.0
    for tp in timing_points:
        if tp.time_ms > time_ms:
            break
        if tp.uninherited:
            current_beat_length_ms = tp.beat_length_ms

    return current_beat_length_ms
