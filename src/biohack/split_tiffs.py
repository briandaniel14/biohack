import os
import uuid
from pathlib import Path

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
    input_file: str,
    output_dir: str,
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
    output_dir = Path(output_dir) / str(uuid.uuid4())
    input_file = Path(input_file)

    brightfield_dir = output_dir / "brightfield/"
    gfp_dir = output_dir / "gfp/"

    os.makedirs(brightfield_dir, exist_ok=True)
    os.makedirs(gfp_dir, exist_ok=True)

    
    
    bf_ch = brightfield_channel_index
    gfp_ch = gfp_channel_index

    stem = input_file.stem
    if verbose:
        print(f"\nProcessing {input_file}")
    img = imread(input_file)
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
