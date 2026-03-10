# Pattern Recognition in Videogames (.osu) - Chunking Pipeline

This branch now focuses on **pattern chunking** instead of pattern naming.

## Objective

Given `.osu` beatmaps, generate per-object training rows that describe:
- chunk boundary membership (combo-based)
- chunk numbering within a pattern (`1,2,3,...`)
- geometric/timing context for model learning

Chunk labels are derived from combo structure:
- `is_new_combo`: boundary flag from hit object type bits
- `chunk_id`: running chunk index in a map
- `chunk_pos`: position inside chunk (on-screen number)

## Current Pipeline

1. Parse `[TimingPoints]` and `[HitObjects]` from `.osu`.
2. Derive combo/chunk labels from `new_combo` boundaries.
3. Extract per-object features (timing, spacing, direction).
4. Add a **6-note history context window** (`i-5 ... i`) with relative placement/timing.
5. Write one consolidated CSV.

## Setup

```powershell
cd "C:\Users\ben20\Desktop\Pain\Projects\Pattern_Recognition_in_Videogames"
pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, run scripts directly with venv python:

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_osu.py --input data\raw --output-dir data\processed
```

## Run Preprocessing

```powershell
python scripts\preprocess_osu.py --input data\raw --output-dir data\processed
```

## Output

- `data/processed/all_objects.csv`

## Important CSV Columns

Core:
- `map_path`, `object_index`, `time_ms`, `x`, `y`
- `dt_ms`, `dt_beats`, `dx`, `dy`, `distance`, `norm_distance`, `turn_angle_deg`
- `is_slider`, `is_spinner`, `is_new_combo`
- `chunk_id`, `chunk_pos`

6-note context window (slot `0..5`, where `5` is current note):
- `ctx_valid_k`
- `ctx_rel_x_k`
- `ctx_rel_y_k`
- `ctx_rel_dt_beats_k`

## Notes

- This branch currently prioritizes preprocessing and labeling for chunk-learning.
- `train_transformer.py` / model training can be aligned to chunk targets next.
