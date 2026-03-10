from typing import Dict, List

# Chunk-learning target columns.
CHUNK_TARGET_COLUMNS: List[str] = ["chunk_id", "chunk_pos"]
CHUNK_FLAG_COLUMN: str = "is_new_combo"


# Optional classification labels for boundary prediction.
BOUNDARY_LABELS: List[str] = ["inside_chunk", "chunk_start"]
BOUNDARY_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(BOUNDARY_LABELS)}
