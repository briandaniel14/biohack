"""Combined Dice + Focal loss for filament segmentation."""

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss from logits. Works on MPS, CUDA, and CPU."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    return ((1 - p_t) ** gamma * bce).mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.gamma = gamma
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dw * self.dice(logits, targets) + self.fw * focal_loss(
            logits, targets, self.gamma
        )
