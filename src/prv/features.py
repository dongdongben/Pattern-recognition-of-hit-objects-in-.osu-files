from __future__ import annotations

from typing import Dict, List

from .osu_parser import ParsedBeatmap


def extract_object_features(parsed: ParsedBeatmap, circle_radius_px: float = 64.0) -> List[Dict[str, float]]:
    """Return per-hit-object features for sequence modeling."""
    raise NotImplementedError("TODO: implement feature extraction")
