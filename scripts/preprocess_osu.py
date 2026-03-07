from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess osu beatmaps into training csv files.")
    parser.add_argument("--input", type=Path, required=True, help="Path to .osu file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    _ = parser.parse_args()

    raise NotImplementedError("TODO: parse .osu files, extract features, label, split, and save csv outputs")


if __name__ == "__main__":
    main()
