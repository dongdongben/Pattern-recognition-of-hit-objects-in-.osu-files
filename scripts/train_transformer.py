from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prv.dataset import FEATURE_COLUMNS, SequenceChunkDataset
from prv.model import OsuPatternTransformer


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    n_correct_l1 = 0
    n_correct_l2 = 0
    n_tokens = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y1 = batch["y1"].to(device)
            y2 = batch["y2"].to(device)
            mask = batch["mask"].to(device)

            logits1, logits2 = model(x, mask)
            pred1 = logits1.argmax(dim=-1)
            pred2 = logits2.argmax(dim=-1)

            valid = mask
            n_correct_l1 += ((pred1 == y1) & valid).sum().item()
            n_correct_l2 += ((pred2 == y2) & valid).sum().item()
            n_tokens += valid.sum().item()

    if n_tokens == 0:
        return 0.0, 0.0
    return n_correct_l1 / n_tokens, n_correct_l2 / n_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer token classifier for osu patterns.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SequenceChunkDataset(str(args.train_csv), seq_len=args.seq_len)
    val_ds = SequenceChunkDataset(str(args.val_csv), seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = OsuPatternTransformer(input_dim=len(FEATURE_COLUMNS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    best_l2 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y1 = batch["y1"].to(device)
            y2 = batch["y2"].to(device)
            mask = batch["mask"].to(device)

            logits1, logits2 = model(x, mask)
            loss1 = loss_fn(logits1.view(-1, logits1.size(-1)), y1.view(-1))
            loss2 = loss_fn(logits2.view(-1, logits2.size(-1)), y2.view(-1))
            loss = loss1 + loss2

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_l1, val_l2 = evaluate(model, val_loader, device)

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_l1_acc={val_l1:.4f} val_l2_acc={val_l2:.4f}"
        )

        if val_l2 > best_l2:
            best_l2 = val_l2
            ckpt = args.artifact_dir / "best_model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seq_len": args.seq_len,
                    "input_dim": len(FEATURE_COLUMNS),
                    "best_val_l2_acc": best_l2,
                },
                ckpt,
            )

    print(f"Training complete. Best Level-2 val acc: {best_l2:.4f}")


if __name__ == "__main__":
    main()
