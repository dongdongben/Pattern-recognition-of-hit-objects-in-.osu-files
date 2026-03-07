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


@dataclass
class ParsedBeatmap:
    path: str
    timing_points: List[TimingPoint]
    hit_objects: List[HitObject]


def parse_osu_file(path: str | Path) -> ParsedBeatmap:
    """Parse [TimingPoints] and [HitObjects] from a .osu file."""
    raise NotImplementedError("TODO: implement .osu parsing")


def get_active_beat_length_ms(timing_points: List[TimingPoint], time_ms: int) -> float:
    """Return the active uninherited beat length at a specific timestamp."""
    raise NotImplementedError("TODO: implement timing point lookup")
