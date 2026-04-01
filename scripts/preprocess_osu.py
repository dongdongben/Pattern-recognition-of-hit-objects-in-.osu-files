from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prv.features import extract_object_features
from prv.heuristics import apply_chunk_labels
from prv.osu_parser import parse_osu_file


def iter_osu_files(input_path: Path):
    if input_path.is_file() and input_path.suffix.lower() == ".osu":
        yield input_path
        return

    if input_path.is_dir():
        yield from sorted(input_path.rglob("*.osu"))

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
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = list(iter_osu_files(args.input))
    if not files:
        raise SystemExit(f"No .osu files found in: {args.input}")

    for file_path in files:
        parsed = parse_osu_file(file_path)
        rows = extract_object_features(parsed)
        rows = apply_chunk_labels(rows)

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
    }
    with (args.output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {args.output_dir / 'all_objects.csv'}")
    print(f"Maps: {len(files)}")
    print(f"Rows: {len(all_rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")


if __name__ == "__main__":
    main()

