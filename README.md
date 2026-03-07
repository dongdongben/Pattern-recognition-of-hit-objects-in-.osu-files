# Pattern Recognition in Videogames (.osu) - Learning Skeleton

This branch is intentionally scaffold-only so you can implement each part yourself.

## Setup

```powershell
cd "C:\Users\ben20\Desktop\Pain\Projects\Pattern_Recognition_in_Videogames"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Files to implement

- `src/prv/osu_parser.py`
- `src/prv/features.py`
- `src/prv/heuristics.py`
- `src/prv/dataset.py`
- `src/prv/model.py`
- `scripts/preprocess_osu.py`
- `scripts/train_transformer.py`

##Functions to implement
parse_osu_file: reads one .osu file and extracts structured timing + hit object sequences.

get_active_beat_length_ms: gives beat length at each object timestamp so timing is beat-normalized (dt_beats), which is crucial for rhythm-aware features.

extract_object_features: converts raw objects into model-ready per-object numeric features (timing gaps, spatial deltas, distance, angle, object type flags).

label_level1: assigns coarse pseudo-label (jumps/streams/sliders/filler) from feature rules.

label_level2: assigns subtype pseudo-label (jump_aim, flow_aim, etc.) conditioned on features and Level-1.

apply_heuristic_labels: runs labeling across a full object sequence and appends both label levels.

SequenceChunkDataset.init: loads CSV, groups by map, chunks into fixed-length sequences for Transformer training.

SequenceChunkDataset.getitem: returns one training sample (x, labels, mask).

PositionalEncoding.init: builds positional signal so the model knows order in sequence.

PositionalEncoding.forward: adds that positional information to token embeddings.

OsuPatternTransformer.init: defines encoder + two classifier heads (Level-1, Level-2).

OsuPatternTransformer.forward: runs sequence through encoder and outputs per-token logits for both label levels.

main in preprocess_osu.py: orchestration step for building processed train/val/test CSVs from raw .osu.

main in train_transformer.py: orchestration step for training loop, validation, and checkpoint saving.


Each file has function signatures and TODO placeholders.
