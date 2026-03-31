from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prv.features import extract_object_features
from prv.heuristics import apply_chunk_labels
from prv.osu_parser import ParsedBeatmap, parse_osu_file


def iter_osu_files(input_path: Path):
    if input_path.is_file() and input_path.suffix.lower() == ".osu":
        yield input_path
        return

    if input_path.is_dir():
        yield from sorted(input_path.rglob("*.osu"))


def normalize_map_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def build_original_lookup(original_input: Path | None) -> dict[str, Path]:
    if original_input is None or not original_input.exists():
        return {}

    lookup: dict[str, Path] = {}
    for path in iter_osu_files(original_input):
        key = normalize_map_name(path.name)
        if key in lookup:
            raise ValueError(f"Duplicate normalized original map name for {path} and {lookup[key]}")
        lookup[key] = path
    return lookup


def find_original_match(raw_path: Path, original_lookup: dict[str, Path]) -> Path | None:
    if not original_lookup:
        return None
    return original_lookup.get(normalize_map_name(raw_path.name))


def attach_original_combo_features(
    rows: list[dict[str, float]],
    chunk_parsed: ParsedBeatmap,
    original_parsed: ParsedBeatmap,
) -> list[dict[str, float]]:
    if len(chunk_parsed.hit_objects) != len(original_parsed.hit_objects):
        raise ValueError(
            f"Hit object count mismatch for {chunk_parsed.path} vs {original_parsed.path}: "
            f"{len(chunk_parsed.hit_objects)} != {len(original_parsed.hit_objects)}"
        )

    for idx, (row, chunk_obj, orig_obj) in enumerate(
        zip(rows, chunk_parsed.hit_objects, original_parsed.hit_objects), start=1
    ):
        same_time = chunk_obj.time_ms == orig_obj.time_ms
        close_x = abs(chunk_obj.x - orig_obj.x) <= 2
        close_y = abs(chunk_obj.y - orig_obj.y) <= 2
        if not (same_time and close_x and close_y):
            raise ValueError(
                f"Hit object mismatch at index {idx} for {chunk_parsed.path} vs {original_parsed.path}: "
                f"chunk=({chunk_obj.time_ms}, {chunk_obj.x}, {chunk_obj.y}) "
                f"orig=({orig_obj.time_ms}, {orig_obj.x}, {orig_obj.y})"
            )

        row["combo_id"] = float(orig_obj.chunk_id)
        row["combo_pos"] = float(orig_obj.chunk_pos)
        row["combo_is_new_combo"] = 1.0 if orig_obj.new_combo else 0.0

    return rows


# takes in the map path, ratio of train val, and seed
def split_map_paths(
    map_paths: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, set[str]]:
    rng = random.Random(seed)
    shuffled = list(map_paths)
    rng.shuffle(shuffled)

    n_maps = len(shuffled)
    n_train = int(n_maps * train_ratio)
    n_val = int(n_maps * val_ratio)

    train_maps = set(shuffled[:n_train])
    val_maps = set(shuffled[n_train : n_train + n_val])
    test_maps = set(shuffled[n_train + n_val :])
    return {"train": train_maps, "val": val_maps, "test": test_maps}


def write_rows(rows: list[dict[str, float]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows: list[dict[str, float]] = []

    parser = argparse.ArgumentParser(description="Preprocess osu beatmaps into chunk-labeled csv files.")
    parser.add_argument("--input", type=Path, required=True, help="Path to chunk-labeled .osu file or directory.")
    parser.add_argument(
        "--original-input",
        type=Path,
        default=Path("data/original"),
        help="Path to original unlabeled .osu files used to recover original combo numbering.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = list(iter_osu_files(args.input))
    if not files:
        raise SystemExit(f"No .osu files found in: {args.input}")

    original_lookup = build_original_lookup(args.original_input)
    matched_original = 0

    for file_path in files:
        parsed = parse_osu_file(file_path)
        rows = extract_object_features(parsed)
        rows = apply_chunk_labels(rows)

        original_path = find_original_match(file_path, original_lookup)
        if original_path is not None:
            original_parsed = parse_osu_file(original_path)
            rows = attach_original_combo_features(rows, parsed, original_parsed)
            matched_original += 1
        else:
            for row in rows:
                row["combo_id"] = 0.0
                row["combo_pos"] = 0.0
                row["combo_is_new_combo"] = 0.0

        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows were produced during preprocessing.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())

    write_rows(all_rows, args.output_dir / "all_objects.csv", fieldnames)

    map_paths = sorted({str(row["map_path"]) for row in all_rows})
    split_maps = split_map_paths(map_paths, args.train_ratio, args.val_ratio, args.seed)

    train_rows = [row for row in all_rows if str(row["map_path"]) in split_maps["train"]]
    val_rows = [row for row in all_rows if str(row["map_path"]) in split_maps["val"]]
    test_rows = [row for row in all_rows if str(row["map_path"]) in split_maps["test"]]

    write_rows(train_rows, args.output_dir / "train.csv", fieldnames)
    write_rows(val_rows, args.output_dir / "val.csv", fieldnames)
    write_rows(test_rows, args.output_dir / "test.csv", fieldnames)

    summary = {
        "n_maps": len(map_paths),
        "n_rows_total": len(all_rows),
        "n_rows_train": len(train_rows),
        "n_rows_val": len(val_rows),
        "n_rows_test": len(test_rows),
        "n_maps_train": len(split_maps["train"]),
        "n_maps_val": len(split_maps["val"]),
        "n_maps_test": len(split_maps["test"]),
        "n_original_maps_matched": matched_original,
        "n_original_maps_available": len(original_lookup),
    }
    with (args.output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {args.output_dir / 'all_objects.csv'}")
    print(f"Maps: {len(files)}")
    print(f"Rows: {len(all_rows)}")
    print(f"Original combo maps matched: {matched_original}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")


if __name__ == "__main__":
    main()

