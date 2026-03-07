from typing import Dict, List

LEVEL1_LABELS: List[str] = ["jumps", "streams", "sliders", "filler"]
LEVEL2_LABELS: List[str] = [
    "jump_aim",
    "awkward_aim",
    "flow_aim",
    "accelerating_flow_aim",
    "slider_aim",
    "linear_aim",
]

LEVEL1_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(LEVEL1_LABELS)}
LEVEL2_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(LEVEL2_LABELS)}
