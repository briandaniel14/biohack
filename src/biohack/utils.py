import numpy as np
from PIL import Image
from IPython.display import display
from pathlib import Path


def load_tiffs(tiff_paths: list[str]) -> list[list[np.array]]:
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
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower()
    )
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

    if arr.ndim != 2:
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
