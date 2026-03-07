from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prv.features import extract_object_features
from prv.heuristics import apply_heuristic_labels
from prv.osu_parser import parse_osu_file


FIELDNAMES = [
    "map_path",
    "object_index",
    "time_ms",
    "x",
    "y",
    "dt_ms",
    "dt_beats",
    "dx",
    "dy",
    "distance",
    "norm_distance",
    "turn_angle_deg",
    "is_slider",
    "is_spinner",
    "label_level1",
    "label_level2",
]


def iter_osu_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".osu":
        yield input_path
        return
    if input_path.is_dir():
        yield from input_path.rglob("*.osu")


def split_map_paths(
    map_paths: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, set[str]]:
    random.seed(seed)
    shuffled = list(map_paths)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = set(shuffled[:n_train])
    val = set(shuffled[n_train : n_train + n_val])
    test = set(shuffled[n_train + n_val :])

    return {"train": train, "val": val, "test": test}


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess osu beatmaps into training csv files.")
    parser.add_argument("--input", type=Path, required=True, help="Path to .osu file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = sorted(iter_osu_files(args.input))
    if not files:
        raise SystemExit(f"No .osu files found in: {args.input}")

    all_rows: List[Dict[str, float]] = []
    for file_path in files:
        parsed = parse_osu_file(file_path)
        feats = extract_object_features(parsed)
        labeled = apply_heuristic_labels(feats)
        all_rows.extend(labeled)

    all_csv = args.output_dir / "all_objects.csv"
    write_csv(all_rows, all_csv)

    map_paths = sorted({str(r["map_path"]) for r in all_rows})
    split = split_map_paths(map_paths, args.train_ratio, args.val_ratio, args.seed)

    train_rows = [r for r in all_rows if str(r["map_path"]) in split["train"]]
    val_rows = [r for r in all_rows if str(r["map_path"]) in split["val"]]
    test_rows = [r for r in all_rows if str(r["map_path"]) in split["test"]]

    write_csv(train_rows, args.output_dir / "train.csv")
    write_csv(val_rows, args.output_dir / "val.csv")
    write_csv(test_rows, args.output_dir / "test.csv")

    summary = {
        "n_maps": len(map_paths),
        "n_rows_total": len(all_rows),
        "n_rows_train": len(train_rows),
        "n_rows_val": len(val_rows),
        "n_rows_test": len(test_rows),
        "n_maps_train": len(split["train"]),
        "n_maps_val": len(split["val"]),
        "n_maps_test": len(split["test"]),
    }
    with (args.output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Preprocessing complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
