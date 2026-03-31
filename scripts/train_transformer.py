from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prv.dataset import FEATURE_COLUMNS, SequenceChunkDataset
from prv.model import OsuPatternTransformer


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    tp = 0
    fp = 0
    fn = 0
    n_correct = 0
    n_tokens = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            logits = model(x, mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            total_loss += loss.item()
            n_batches += 1

            pred = logits.argmax(dim=-1)
            valid = mask
            n_correct += ((pred == y) & valid).sum().item()
            n_tokens += valid.sum().item()
            tp += ((pred == 1) & (y == 1) & valid).sum().item()
            fp += ((pred == 1) & (y == 0) & valid).sum().item()
            fn += ((pred == 0) & (y == 1) & valid).sum().item()

    avg_loss = total_loss / max(1, n_batches)
    accuracy = n_correct / max(1, n_tokens)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return avg_loss, accuracy, precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer token classifier for osu chunk starts.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=100)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--overfit-one-batch", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SequenceChunkDataset(str(args.train_csv), seq_len=args.seq_len)
    val_ds = SequenceChunkDataset(str(args.val_csv), seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = OsuPatternTransformer(
        input_dim=len(FEATURE_COLUMNS),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    debug_batch = next(iter(train_loader)) if args.overfit_one_batch else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        batch_iterable = [debug_batch] if debug_batch is not None else train_loader

        for batch in batch_iterable:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            logits = model(x, mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_loss, val_acc, val_prec, val_rec = evaluate(model, val_loader, loss_fn, device)

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_chunk_start_acc={val_acc:.4f} "
            f"val_chunk_start_prec={val_prec:.4f} val_chunk_start_rec={val_rec:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            ckpt = args.artifact_dir / "best_model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": len(FEATURE_COLUMNS),
                    "num_classes": 2,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "target": "chunk_start",
                    "feature_columns": FEATURE_COLUMNS,
                    "hparams": {
                        "seq_len": args.seq_len,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "num_layers": args.num_layers,
                        "dim_feedforward": args.dim_feedforward,
                        "grad_clip": args.grad_clip,
                    },
                },
                ckpt,
            )
        else:
            epochs_without_improvement += 1
            if not args.overfit_one_batch and epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch was {best_epoch} with val_loss={best_val_loss:.4f}"
                )
                break

    print(f"Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
