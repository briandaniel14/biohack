"""
Basic training script — Stage 4.
Trains a U-Net on synthetic data, evaluates on real annotated data.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --epochs 20 --batch_size 32
"""

import argparse
import os
import sys

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from biohack.dataset import RealDataset, SyntheticDataset
from biohack.loss import DiceFocalLoss
from biohack.model import build_unet

sys.path.insert(0, os.path.dirname(__file__))
from eval_metrics import evaluate  # noqa: E402

THRESHOLD = 0.5

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


def make_weighted_sampler(dataset: SyntheticDataset) -> WeightedRandomSampler:
    """Oversample positive (filament) images so each batch is ~50/50."""
    print("  Building sampler weights...")
    has_filament = np.array(
        [tifffile.imread(os.path.join(dataset.mask_dir, f)).any() for f in dataset.files],  # noqa: PERF401
        dtype=np.float32,
    )
    pos = has_filament.sum()
    neg = len(has_filament) - pos
    print(f"  Positives: {int(pos)}  Negatives: {int(neg)}")
    w_pos = 1.0 / pos if pos else 0
    w_neg = 1.0 / neg if neg else 0
    weights = np.where(has_filament, w_pos, w_neg).tolist()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:  # noqa: PLW2901
        imgs = imgs.to(DEVICE)  # noqa: PLW2901
        masks = masks.to(DEVICE)  # noqa: PLW2901
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_on_real(model, dataset: RealDataset) -> dict:
    model.eval()
    preds, targets = [], []
    loader = DataLoader(dataset, batch_size=32)
    for imgs, masks in loader:  # noqa: PLW2901
        imgs = imgs.to(DEVICE)  # noqa: PLW2901
        logits = model(imgs)
        pred = (torch.sigmoid(logits) > THRESHOLD).cpu().numpy().squeeze(1).astype(bool)
        tgt = masks.numpy().squeeze(1).astype(bool)
        preds.extend(list(pred))
        targets.extend(list(tgt))
    return evaluate(preds, targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", default="synthetic_data")
    parser.add_argument("--test_pkl", default="data/annotated_data.pkl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--output", default="outputs/model.pt")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training set size")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    train_ds = SyntheticDataset(args.synthetic_dir)
    if args.max_samples:
        train_ds.files = train_ds.files[: args.max_samples]
    test_ds = RealDataset(args.test_pkl)
    print(f"Train: {len(train_ds)} synthetic  |  Test: {len(test_ds)} real")

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)

    model = build_unet(encoder=args.encoder).to(DEVICE)
    loss_fn = DiceFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        metrics = eval_on_real(model, test_ds)
        scheduler.step()

        print(
            f"Epoch {epoch:>2}/{args.epochs}  "
            f"loss={train_loss:.4f}  "
            f"dice={metrics['dice']:.4f}  "
            f"iou={metrics['iou']:.4f}"
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(model.state_dict(), args.output)

    print(f"\nBest dice: {best_dice:.4f}  →  saved to {args.output}")


if __name__ == "__main__":
    main()
