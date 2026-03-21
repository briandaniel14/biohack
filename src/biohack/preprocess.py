"""
Image preprocessing for filament segmentation.
Handles both real (uint16) and synthetic (uint8) data.
"""

import numpy as np


def preprocess(image: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    """
    Preprocess a single grayscale image for model input.

    Steps:
      1. Cast to float32
      2. Clip hot pixels at clip_percentile
      3. Per-image z-score normalisation

    Args:
        image: 2D array, any dtype (uint8 or uint16)
        clip_percentile: upper percentile to clip before normalising

    Returns:
        float32 array of same shape, z-score normalised
    """
    img = image.astype(np.float32)
    upper = np.percentile(img, clip_percentile)
    img = np.clip(img, img.min(), upper)
    mean, std = img.mean(), img.std()
    min_std = 1e-6
    if std < min_std:
        return np.zeros_like(img)
    return (img - mean) / std


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Convert any mask representation to a bool array."""
    return mask.astype(bool)
