# Pattern Recognition in Videogames (.osu) - Chunking Pipeline

This branch focuses on **chunking hit-object sequences** in `.osu` beatmaps rather than assigning pattern names.

## Objective

Given a beatmap, the pipeline builds one row per hit object and trains models to predict:
- chunk boundaries
- chunk position inside a chunk (`chunk_pos = 1, 2, 3, ...`)

Chunk labels are currently derived from combo structure in the `.osu` file:
- `is_new_combo`: combo boundary flag from hit object type bits
- `chunk_id`: running chunk index in a map
- `chunk_pos`: position inside the current chunk

The current modeling goal is to learn chunk structure from **geometry and timing**, not by directly feeding combo-boundary metadata into the model.

## Current Pipeline

1. Parse `[TimingPoints]` and `[HitObjects]` from `.osu` files.
2. Derive chunk labels from combo boundaries.
3. Extract per-object geometric and temporal features.
4. Write one consolidated CSV plus map-level train/val/test splits.
5. Train either:
   - a logistic chunk-boundary baseline, or
   - a Transformer token classifier for `chunk_pos`.

## Current Feature Set

The Transformer currently uses the following input columns from `src/prv/dataset.py`:
- `x`
- `y`
- `dt_ms`
- `dt_beats`
- `dx`
- `dy`
- `distance`
- `norm_distance`
- `is_slider`
- `is_spinner`

Notably, `is_new_combo` is still written to the CSV for analysis, but it is **not** fed into the current Transformer input.

## Preprocessing

Run preprocessing from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_osu.py --input data\raw --output-dir data\processed
```

This writes:
- `data/processed/all_objects.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/split_summary.json`

## Important CSV Columns

Metadata / labels:
- `map_path`
- `object_index`
- `time_ms`
- `chunk_id`
- `chunk_pos`
- `is_new_combo`

Geometry / timing:
- `x`
- `y`
- `dt_ms`
- `dt_beats`
- `dx`
- `dy`
- `distance`
- `norm_distance`
- `is_slider`
- `is_spinner`

## Baseline Model

A simple baseline is implemented in `scripts/train_baseline.py`.

It trains a linear chunk-boundary classifier (logistic regression style), predicts whether each object starts a new chunk, and reconstructs `chunk_pos` from predicted boundaries.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py --csv data\processed\all_objects.csv --output-dir data\baseline
```

Current reported baseline metrics on the held-out test split:
- boundary accuracy: `61.19%`
- boundary precision: `28.36%`
- boundary recall: `65.50%`
- boundary F1: `0.3958`
- `chunk_pos` exact match: `22.80%`
- `chunk_pos` MAE: `2.1839`

## Transformer Model

The main sequence model is implemented in:
- `src/prv/model.py`
- `scripts/train_transformer.py`

Model notes:
- linear input projection from numeric note features
- sinusoidal positional encoding
- multi-head self-attention encoder blocks
- **dropout removed** from the current Transformer implementation

Run training:

```powershell
.\.venv\Scripts\python.exe scripts\train_transformer.py --train-csv data\processed\train.csv --val-csv data\processed\val.csv
```

Useful arguments:
- `--seq-len`
- `--batch-size`
- `--lr`
- `--d-model`
- `--nhead`
- `--num-layers`
- `--dim-feedforward`
- `--overfit-one-batch`

Example overfit sanity check:

```powershell
.\.venv\Scripts\python.exe scripts\train_transformer.py --train-csv data\processed\one_map_train.csv --val-csv data\processed\one_map_train.csv --epochs 200 --batch-size 1 --lr 1e-3 --d-model 256 --nhead 4 --num-layers 4 --dim-feedforward 512 --overfit-one-batch
```

## Current Findings

- The Transformer can overfit a single batch and a single map, which confirmed that the training path, masking, and label handling are working.
- On the full map-level split, performance is much stronger when `is_new_combo` is included as an input than when it is excluded.
- After removing `is_new_combo`, full-split validation accuracy drops substantially, which suggests the current task is genuinely difficult from geometry/timing alone.
- This makes the current project more faithful to perceptual chunking, but also means more maps and stronger cross-map generalization experiments are needed.

## Notes

- The current repo state is centered on chunking, not pattern-type classification.
- The main open challenge is improving cross-map generalization without giving the model direct combo-boundary shortcuts.
