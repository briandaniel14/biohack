from pathlib import Path

import numpy as np
from IPython.display import display
from PIL import Image


def load_tiffs(tiff_paths: list[Path]) -> list[list[np.ndarray]]:
    movies = []

    for path in tiff_paths:
        img = Image.open(path)
        frames = []

        try:
            while True:
                frames.append(np.array(img))
                img.seek(img.tell() + 1)
        except EOFError:
            pass  # No more frames

        movies.append(frames)

    return movies


def load_image_paths(image_dir: str | Path) -> list[Path]:
    image_dir = Path(image_dir)
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower())
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    return paths


def load_grayscale_image(path: str | Path, convert: str = "L") -> np.ndarray:
    img = Image.open(path).convert(convert)
    return np.array(img)


def matrix_to_image(
    matrix: np.ndarray,
    mode: str = "L",
    normalize: bool = True,
    show: bool = True,
) -> Image.Image:
    """Convert a 2D NumPy matrix to a PIL image and optionally display it.

    Args:
        matrix: 2D NumPy array.
        mode: PIL mode ("L" for grayscale, "RGB" for color).
        normalize: If True, scales values to [0, 255].
        show: If True, displays image inline in notebook.

    Returns:
        PIL.Image.Image object.
    """
    arr = np.asarray(matrix)

    if arr.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}")

    if normalize:
        arr = arr.astype(np.float32)
        min_val = arr.min()
        max_val = arr.max()
        if max_val == min_val:
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = ((arr - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode=mode)

    if show:
        display(img)

    return img



def threshold_for_black_below_percent(
    image: np.ndarray,
    black_percent: float,
    normalized: bool = True,
) -> float:
    """Compute a threshold that marks the lowest n% of pixels as black.

    Args:
        image: 2D image matrix.
        black_percent: Percentage of darkest pixels to set to black, in [0, 100].
        normalized: If True, return threshold in [0, 1]; otherwise return in
            the original image intensity scale.

    Returns:
        Intensity threshold value.
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {arr.shape}")
    if not 0 <= black_percent <= 100:
        raise ValueError(f"black_percent must be in [0, 100], got {black_percent}")

    arr_float = arr.astype(np.float32)
    min_val = float(arr_float.min())
    max_val = float(arr_float.max())
    if max_val == min_val:
        return 0.0 if normalized else min_val

    arr_normalized = (arr_float - min_val) / (max_val - min_val)
    threshold_normalized = float(np.percentile(arr_normalized, black_percent))

    if normalized:
        return threshold_normalized
    return threshold_normalized * (max_val - min_val) + min_val


def set_below_threshold_to_black(image: np.ndarray, threshold: float) -> np.ndarray:
    """Set pixels below or equal to a threshold to black (0).

    The threshold is interpreted in normalized intensity space ([0, 1]) when
    threshold <= 1. If threshold > 1, it is interpreted in the image's original
    intensity scale and converted to normalized intensity using image min/max.
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {arr.shape}")

    arr_float = arr.astype(np.float32)
    min_val = float(arr_float.min())
    max_val = float(arr_float.max())
    if max_val == min_val:
        return np.zeros_like(arr)

    arr_normalized = (arr_float - min_val) / (max_val - min_val)
    threshold_normalized = float(threshold)
    if threshold_normalized > 1.0:
        threshold_normalized = (threshold_normalized - min_val) / (max_val - min_val)
    threshold_normalized = float(np.clip(threshold_normalized, 0.0, 1.0))

    result = arr.copy()
    result[arr_normalized <= threshold_normalized] = 0
    return result


def set_lowest_percent_to_black(image: np.ndarray, black_percent: float) -> np.ndarray:
    """Set the darkest n% of pixels to black (0)."""
    threshold = threshold_for_black_below_percent(
        image=image,
        black_percent=black_percent,
        normalized=True,
    )
    return set_below_threshold_to_black(image=image, threshold=threshold)


def threshold_for_remove_below_percent(
    image: np.ndarray,
    remove_percent: float,
    normalized: bool = True,
) -> float:
    """Backward-compatible alias for threshold_for_black_below_percent."""
    return threshold_for_black_below_percent(
        image=image,
        black_percent=remove_percent,
        normalized=normalized,
    )
