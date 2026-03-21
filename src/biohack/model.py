"""U-Net model for filament segmentation."""

import segmentation_models_pytorch as smp
import torch
from torch import nn


def build_unet(encoder: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """
    Small U-Net with a lightweight encoder.
    Input:  (B, 1, 128, 128) float32
    Output: (B, 1, 128, 128) float32 logits
    """
    weights = "imagenet" if pretrained else None
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=1,
        classes=1,
        activation=None,  # raw logits — loss functions handle sigmoid
    )


def predict_mask(model: nn.Module, image: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Run inference and return a boolean mask.

    Args:
        model: trained U-Net
        image: (1, H, W) or (B, 1, H, W) float32 tensor
        threshold: sigmoid threshold for positive class

    Returns:
        bool tensor, same spatial dims as input
    """
    if image.dim() == 3:  # noqa: PLR2004
        image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(image)
        return (torch.sigmoid(logits) > threshold).squeeze()
