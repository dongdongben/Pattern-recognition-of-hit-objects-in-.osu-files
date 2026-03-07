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

    @property
    def is_slider(self) -> bool:
        return bool(self.obj_type & 2)

    @property
    def is_spinner(self) -> bool:
        return bool(self.obj_type & 8)


@dataclass
class ParsedBeatmap:
    path: str
    timing_points: List[TimingPoint]
    hit_objects: List[HitObject]


def _parse_timing_point(line: str) -> TimingPoint | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return None
    try:
        uninherited = parts[6] == "1"
        return TimingPoint(
            time_ms=float(parts[0]),
            beat_length_ms=float(parts[1]),
            meter=int(parts[2]),
            uninherited=uninherited,
        )
    except ValueError:
        return None


def _parse_hit_object(line: str) -> HitObject | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return None
    try:
        raw_params = ",".join(parts[5:]) if len(parts) > 5 else ""
        return HitObject(
            x=int(parts[0]),
            y=int(parts[1]),
            time_ms=int(parts[2]),
            obj_type=int(parts[3]),
            raw_params=raw_params,
        )
    except ValueError:
        return None


def parse_osu_file(path: str | Path) -> ParsedBeatmap:
    path = Path(path)
    current_section = None
    timing_points: List[TimingPoint] = []
    hit_objects: List[HitObject] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line
                continue
            if current_section == "[TimingPoints]":
                tp = _parse_timing_point(line)
                if tp is not None:
                    timing_points.append(tp)
            elif current_section == "[HitObjects]":
                obj = _parse_hit_object(line)
                if obj is not None:
                    hit_objects.append(obj)

    timing_points.sort(key=lambda t: t.time_ms)
    hit_objects.sort(key=lambda h: h.time_ms)

    return ParsedBeatmap(
        path=str(path),
        timing_points=timing_points,
        hit_objects=hit_objects,
    )


def get_active_beat_length_ms(timing_points: List[TimingPoint], time_ms: int) -> float:
    # Fallback to 120 BPM if no uninherited timing point is available.
    if not timing_points:
        return 500.0

    current = 500.0
    for tp in timing_points:
        if tp.uninherited and tp.time_ms <= time_ms:
            current = tp.beat_length_ms
        if tp.time_ms > time_ms:
            break
    return current
