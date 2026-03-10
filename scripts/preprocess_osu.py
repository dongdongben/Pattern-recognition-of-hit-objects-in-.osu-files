from __future__ import annotations

import argparse
import csv
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
        yield from input_path.rglob("*.osu")


def main() -> None:
    all_rows = []

    parser = argparse.ArgumentParser(description="Preprocess osu beatmaps into chunk-labeled csv files.")
    parser.add_argument("--input", type=Path, required=True, help="Path to .osu file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    files = list(iter_osu_files(args.input))
    if not files:
        raise SystemExit(f"No .osu files found in: {args.input}")

    for file_path in files:
        parsed = parse_osu_file(file_path)
        rows = extract_object_features(parsed)
        rows = apply_chunk_labels(rows)
        all_rows.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "all_objects.csv"

    fieldnames = list(all_rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote: {out_csv}")
    print(f"Maps: {len(files)}")
    print(f"Rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
