"""
Evaluation metrics for filament segmentation.
Both pred and target should be boolean numpy arrays of shape (128, 128).
"""

import numpy as np


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = (pred & target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0  # both empty = perfect
    return 2 * intersection / denom


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 1.0  # both empty = perfect
    return intersection / union


def evaluate(preds: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    """
    Args:
        preds: list of predicted boolean masks
        targets: list of ground truth boolean masks
    Returns:
        dict with mean Dice, mean IoU, and per-sample scores
    """
    dice_scores = [dice(p, t) for p, t in zip(preds, targets, strict=True)]
    iou_scores = [iou(p, t) for p, t in zip(preds, targets, strict=True)]
    return {
        "dice": np.mean(dice_scores),
        "iou": np.mean(iou_scores),
        "per_sample_dice": dice_scores,
        "per_sample_iou": iou_scores,
    }


if __name__ == "__main__":
    # Quick sanity check
    perfect = np.zeros((128, 128), dtype=bool)
    perfect[60:70, 60:70] = True

    print("Perfect prediction:")
    print(f"  Dice: {dice(perfect, perfect):.4f}")
    print(f"  IoU:  {iou(perfect, perfect):.4f}")

    half_overlap = np.zeros((128, 128), dtype=bool)
    half_overlap[60:70, 65:75] = True  # 50% overlap
    print("\n50% overlap prediction:")
    print(f"  Dice: {dice(half_overlap, perfect):.4f}")
    print(f"  IoU:  {iou(half_overlap, perfect):.4f}")

    empty = np.zeros((128, 128), dtype=bool)
    print("\nBoth empty (no filament):")
    print(f"  Dice: {dice(empty, empty):.4f}")
    print(f"  IoU:  {iou(empty, empty):.4f}")
