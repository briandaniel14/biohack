import os

import numpy as np
from tifffile import imread, imwrite


def split_stack(img: np.ndarray, *, verbose: bool = False) -> np.ndarray:
    """Normalize stack to ``(T, C, Y, X)`` with ``C == 2`` (brightfield + GFP)."""
    if verbose:
        print("Original shape:", img.shape)

    if img.ndim == 4 and img.shape[1] == 2:
        pass
    elif img.ndim == 4 and img.shape[-1] == 2:
        img = np.transpose(img, (0, 3, 1, 2))
    elif img.ndim == 3:
        z, y, x = img.shape
        if z % 2 != 0:
            raise ValueError(
                f"Expected even number of z-slices for interleaved 2-channel data, got {z}"
            )
        t = z // 2
        img = img.reshape(t, 2, y, x)
    else:
        raise ValueError(f"Unsupported shape {img.shape}")

    if verbose:
        print("Converted to:", img.shape, "(T, C, Y, X)")
    return img


def split_two_channel_time_stacks(
    input_dir: str,
    brightfield_dir: str,
    gfp_dir: str,
    *,
    brightfield_channel_index: int = 0,
    gfp_channel_index: int = 1,
    verbose: bool = True,
) -> None:
    """
    Read each TIFF in ``input_dir``, convert with :func:`split_stack`, write BF/GFP frames.

    Parameters
    ----------
    input_dir
        Directory containing raw multi-frame 2-channel ``.tif`` / ``.tiff`` files.
    brightfield_dir, gfp_dir
        Output directories (created if missing).
    brightfield_channel_index, gfp_channel_index
        Which channel index is BF vs GFP after ``(T, C, Y, X)`` layout (default 0 and 1).
    verbose
        Print progress.
    """
    os.makedirs(brightfield_dir, exist_ok=True)
    os.makedirs(gfp_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff"))]
    if not files:
        raise RuntimeError(f"No TIFF files found in {input_dir}")

    bf_ch = brightfield_channel_index
    gfp_ch = gfp_channel_index

    for file in files:
        path = os.path.join(input_dir, file)
        stem = os.path.splitext(file)[0]
        if verbose:
            print(f"\nProcessing {file}")
        img = imread(path)
        img = split_stack(img, verbose=verbose)
        t_n, c_n, _, _ = img.shape
        if c_n < max(bf_ch, gfp_ch) + 1:
            raise ValueError(
                f"Stack has {c_n} channels; need indices {bf_ch}, {gfp_ch}"
            )

        for t in range(t_n):
            bf = img[t, bf_ch]
            gfp = img[t, gfp_ch]
            bf_out = os.path.join(brightfield_dir, f"{stem}_frame_{t + 1:04d}_BF.tif")
            gfp_out = os.path.join(gfp_dir, f"{stem}_frame_{t + 1:04d}_GFP.tif")
            imwrite(bf_out, bf)
            imwrite(gfp_out, gfp)

        if verbose:
            print(f"Wrote {t_n} BF frames to {brightfield_dir}")
            print(f"Wrote {t_n} GFP frames to {gfp_dir}")

    if verbose:
        print("Done")
