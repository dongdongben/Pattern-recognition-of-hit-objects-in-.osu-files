from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
#& "C:\Users\ben20\Desktop\Pain\Projects\Pattern_Recognition_in_Videogames\.venv\Scripts\python.exe" scripts\train_baseline.py --csv data\processed_ctx\all_objects.csv --output-dir data\baseline



ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


EXCLUDED_COLUMNS = {
    "map_path",
    "object_index",
    "time_ms",
    "dx",
    "dy",
    "distance",
    "norm_distance",
    "is_new_combo",
    "chunk_id",
    "chunk_pos",
    "combo_id",
    "combo_pos",
    "combo_is_new_combo",
}


@dataclass
class RowSample:
    map_path: str
    object_index: int
    features: List[float]
    label: float
    true_chunk_pos: int


class LogisticBoundaryModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple chunk-boundary baseline without using is_new_combo as an input feature."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/processed/all_objects.csv"),
        help="Path to preprocessed all_objects.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/baseline"),
        help="Directory for metrics and predictions",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=360)
    return parser.parse_args()


def load_samples(csv_path: Path) -> tuple[List[RowSample], List[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")

        feature_columns = [name for name in reader.fieldnames if name not in EXCLUDED_COLUMNS]
        samples: List[RowSample] = []
        for row in reader:
            features = [float(row[name]) for name in feature_columns]
            sample = RowSample(
                map_path=row["map_path"],
                object_index=int(float(row["object_index"])),
                features=features,
                label=1.0 if float(row["chunk_pos"]) <= 1.0 else 0.0,
                true_chunk_pos=int(float(row["chunk_pos"])),
            )
            samples.append(sample)
    if not samples:
        raise ValueError(f"No rows found in {csv_path}")
    return samples, feature_columns


def split_by_map(samples: Sequence[RowSample], seed: int) -> tuple[List[RowSample], List[RowSample], List[RowSample]]:
    maps = sorted({sample.map_path for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(maps)

    n_maps = len(maps)
    train_end = max(1, int(round(n_maps * 0.67)))
    val_end = max(train_end + 1, int(round(n_maps * 0.83))) if n_maps >= 3 else n_maps
    val_end = min(val_end, n_maps)

    train_maps = set(maps[:train_end])
    val_maps = set(maps[train_end:val_end])
    test_maps = set(maps[val_end:])

    if not val_maps and test_maps:
        moved = next(iter(test_maps))
        test_maps.remove(moved)
        val_maps.add(moved)
    if not test_maps and val_maps:
        moved = next(iter(val_maps))
        val_maps.remove(moved)
        test_maps.add(moved)

    train = [s for s in samples if s.map_path in train_maps]
    val = [s for s in samples if s.map_path in val_maps]
    test = [s for s in samples if s.map_path in test_maps]

    if not train or not val or not test:
        raise ValueError(
            f"Need non-empty train/val/test splits, got train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return train, val, test


def to_tensor(samples: Sequence[RowSample]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([sample.features for sample in samples], dtype=torch.float32)
    y = torch.tensor([sample.label for sample in samples], dtype=torch.float32)
    return x, y


def standardize(
    train_x: torch.Tensor, other_x: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    train_scaled = (train_x - mean) / std
    other_scaled = [(x - mean) / std for x in other_x]
    return train_scaled, other_scaled, mean, std


def compute_class_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum().item()
    negatives = float(labels.numel()) - positives
    if positives <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def binary_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = float(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = float(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = float(((y_true == 1) & (y_pred == 0)).sum().item())
    total = max(1.0, tp + tn + fp + fn)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    accuracy = (tp + tn) / total
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def reconstruct_chunk_positions(samples: Sequence[RowSample], predicted_boundaries: Sequence[int]) -> List[int]:
    predicted_positions: List[int] = []
    current_map = None
    current_pos = 0

    for sample, boundary in zip(samples, predicted_boundaries):
        starts_new_chunk = boundary == 1 or sample.map_path != current_map or sample.object_index == 1
        if starts_new_chunk:
            current_pos = 1
            current_map = sample.map_path
        else:
            current_pos += 1
        predicted_positions.append(current_pos)

    return predicted_positions


def chunk_position_metrics(samples: Sequence[RowSample], predicted_positions: Sequence[int]) -> Dict[str, float]:
    exact = 0
    mae_total = 0.0
    for sample, pred in zip(samples, predicted_positions):
        if pred == sample.true_chunk_pos:
            exact += 1
        mae_total += abs(pred - sample.true_chunk_pos)
    total = max(1, len(predicted_positions))
    return {
        "chunk_pos_exact_match": exact / total,
        "chunk_pos_mae": mae_total / total,
    }


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    samples: Sequence[RowSample],
    threshold: float,
) -> tuple[Dict[str, float], List[float], List[int], List[int]]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).to(torch.int64)

    boundary = binary_metrics(y.to(torch.int64), preds)
    pred_positions = reconstruct_chunk_positions(samples, preds.tolist())
    chunk = chunk_position_metrics(samples, pred_positions)
    metrics = {**boundary, **chunk}
    return metrics, probs.tolist(), preds.tolist(), pred_positions


def save_predictions(
    output_path: Path,
    samples: Sequence[RowSample],
    probs: Sequence[float],
    preds: Sequence[int],
    pred_positions: Sequence[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "map_path",
                "object_index",
                "true_boundary",
                "pred_boundary_prob",
                "pred_boundary",
                "true_chunk_pos",
                "pred_chunk_pos",
            ]
        )
        for sample, prob, pred, pred_pos in zip(samples, probs, preds, pred_positions):
            writer.writerow(
                [
                    sample.map_path,
                    sample.object_index,
                    int(sample.label),
                    f"{prob:.6f}",
                    pred,
                    sample.true_chunk_pos,
                    pred_pos,
                ]
            )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples, feature_columns = load_samples(args.csv)
    train_samples, val_samples, test_samples = split_by_map(samples, seed=args.seed)

    train_x, train_y = to_tensor(train_samples)
    val_x, val_y = to_tensor(val_samples)
    test_x, test_y = to_tensor(test_samples)
    train_x, [val_x, test_x], _, _ = standardize(train_x, [val_x, test_x])

    model = LogisticBoundaryModel(input_dim=train_x.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_class_weight(train_y))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_f1 = -math.inf
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        val_metrics, _, _, _ = evaluate(model, val_x, val_y, val_samples, threshold=args.threshold)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)

    train_metrics, _, _, _ = evaluate(model, train_x, train_y, train_samples, threshold=args.threshold)
    val_metrics, _, _, _ = evaluate(model, val_x, val_y, val_samples, threshold=args.threshold)
    test_metrics, test_probs, test_preds, test_chunk_pos = evaluate(
        model, test_x, test_y, test_samples, threshold=args.threshold
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "csv": str(args.csv),
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "splits": {
            "train_rows": len(train_samples),
            "val_rows": len(val_samples),
            "test_rows": len(test_samples),
            "train_maps": len({s.map_path for s in train_samples}),
            "val_maps": len({s.map_path for s in val_samples}),
            "test_maps": len({s.map_path for s in test_samples}),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path = output_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_predictions(output_dir / "baseline_test_predictions.csv", test_samples, test_probs, test_preds, test_chunk_pos)

    print("Baseline training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Metrics: {metrics_path}")
    print(f"Test predictions: {output_dir / 'baseline_test_predictions.csv'}")
    print("Test boundary metrics:")
    print(
        "  accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}".format(
            **test_metrics
        )
    )
    print("Test chunk metrics:")
    print(
        "  chunk_pos_exact_match={chunk_pos_exact_match:.4f} chunk_pos_mae={chunk_pos_mae:.4f}".format(
            **test_metrics
        )
    )


if __name__ == "__main__":
    main()


