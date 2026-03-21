"""
Synthetic Labelled Data Generator for URA7/URA8 Yeast Cell Filament Segmentation
=================================================================================
Generates paired (image, filament_mask) .tif files for fine-tuning segmentation models.

Three-component image model:
  1. Background noise  — spatially-correlated Gaussian matching real sensor noise
  2. Cell bodies       — soft elliptical blobs with realistic morphology
  3. Filaments         — thin curved structures inside cells (segmentation target)

Calibrated from 64 manually annotated masks across 4 crops:
  ch20_URA7_URA8_001-crop1 (14 masks), 001-crop2 (18), 001-crop3 (23), 002-crop2 (9)
  400 total frames at 128x128, 16-bit fluorescence microscopy
"""

import argparse
import os

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse, line_aa

IMG_SIZE = 128

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — calibrated from 400 real frames + 64 ground-truth masks
# ─────────────────────────────────────────────────────────────────────────────

# Background (corner regions across 400 frames: mean=123.0±2.5, pixel std=5.9±0.3)
BG_MEAN_RANGE = (119, 132)  # covers observed mean range [119.1, 136.4], centered lower
BG_NOISE_STD = 7.0  # raw noise before spatial correlation
BG_SPATIAL_SIGMA = 0.4  # light smoothing to preserve high-freq noise
BG_ILLUMINATION_STD = 2.5  # low-freq uneven illumination

# Cells (from regionprops + intensity analysis)
#   2–6 cells per frame in loose vertical cluster
#   Cell halo: +14 to +37 above background (mean 23.6)
CELLS_PER_IMAGE = (2, 6)
CELL_SEMI_MAJOR = (10, 18)  # semi-major axis (px) — enlarged to host long filaments
CELL_SEMI_MINOR = (8, 15)  # semi-minor axis (px)
CELL_ECCENTRICITY_MEAN = 0.45
CELL_ECCENTRICITY_STD = 0.15
CELL_HALO_INTENSITY = (14, 37)  # peak above background
CELL_EDGE_BLUR = (2.5, 4.0)  # Gaussian sigma for soft edges
CELL_TEXTURE_STD_FRAC = 0.06  # internal granularity

# Filaments (from 64 ground-truth masks)
#   Area: 7–144 px, mean=37.6, median=26  (heavy right tail → lognormal length)
#   Eccentricity: 0.38–0.99, mean=0.911   (very elongated)
#   Major axis: 3.0–31.8 px, mean=13.5    (length)
#   Minor axis: 2.1–7.6 px, mean=3.6      (width)
#   Always exactly 1 connected component per mask
#   Intensity: +28 to +124 above cell body, mean=54.9
FILAMENT_PROB = 0.50
FILAMENT_LENGTH_MU = 2.4  # lognormal mu for spine length
FILAMENT_LENGTH_SIGMA = 0.85  # lognormal sigma (wide spread for heavy tail)
FILAMENT_LENGTH_CLIP = (3, 55)  # hard clip on spine length
FILAMENT_WIDTH_BASE = (0.45, 0.9)  # base Gaussian cross-section sigma
FILAMENT_WIDTH_LENGTH_COEF = 0.018  # width increases with length
FILAMENT_WIDTH_CLIP = (0.45, 1.8)  # hard clip → minor axis ≤ ~7.6 px
FILAMENT_MASK_THRESHOLD = 0.16  # binary mask threshold (tuned to match area dist)
FILAMENT_MIN_AREA = 5  # discard masks smaller than this
FILAMENT_INTENSITY = (28, 125)  # added above cell body
FILAMENT_CURVATURE = (0.01, 0.15)  # angular change per step (low → high eccentricity)
FILAMENT_BRANCH_PROB = 0.08


# ─────────────────────────────────────────────────────────────────────────────
# Component 1: Background noise
# ─────────────────────────────────────────────────────────────────────────────


def generate_background(size=IMG_SIZE, rng=None):
    """Spatially-correlated Gaussian background matching real sensor noise."""
    if rng is None:
        rng = np.random.default_rng()

    bg_mean = rng.uniform(*BG_MEAN_RANGE)
    noise = rng.normal(0, BG_NOISE_STD, (size, size))
    noise = gaussian_filter(noise, sigma=BG_SPATIAL_SIGMA)

    illum = rng.normal(0, BG_ILLUMINATION_STD, (size, size))
    illum = gaussian_filter(illum, sigma=size / 3.5)

    return bg_mean + noise + illum


# ─────────────────────────────────────────────────────────────────────────────
# Component 2: Cell body skeletons
# ─────────────────────────────────────────────────────────────────────────────


def sample_cell_params(n_cells, size=IMG_SIZE, rng=None):
    """Sample cell morphology, arranged in a loose vertical cluster."""
    if rng is None:
        rng = np.random.default_rng()

    cells = []
    cluster_cy = size // 2 + rng.normal(0, 10)
    cluster_cx = size // 2 + rng.normal(0, 8)

    for i in range(n_cells):
        cy = cluster_cy + (i - n_cells / 2) * rng.uniform(10, 18) + rng.normal(0, 4)
        cx = cluster_cx + rng.normal(0, 6)

        ecc = np.clip(rng.normal(CELL_ECCENTRICITY_MEAN, CELL_ECCENTRICITY_STD), 0.05, 0.80)
        semi_major = rng.uniform(*CELL_SEMI_MAJOR)
        semi_minor = semi_major * np.sqrt(1 - ecc**2)
        semi_minor = np.clip(semi_minor, *CELL_SEMI_MINOR)

        angle = rng.uniform(0, np.pi)
        peak = rng.uniform(*CELL_HALO_INTENSITY)
        blur = rng.uniform(*CELL_EDGE_BLUR)

        cells.append(
            dict(  # noqa: C408
                cy=cy,
                cx=cx,
                semi_major=semi_major,
                semi_minor=semi_minor,
                angle=angle,
                peak=peak,
                blur=blur,
            )
        )
    return cells


def render_cell_body(canvas, cell, rng=None):
    """Render a soft elliptical cell body onto canvas (additive)."""
    if rng is None:
        rng = np.random.default_rng()
    size = canvas.shape[0]

    rr, cc = ellipse(
        int(cell["cy"]),
        int(cell["cx"]),
        int(cell["semi_major"]),
        int(cell["semi_minor"]),
        shape=(size, size),
        rotation=cell["angle"],
    )

    cell_layer = np.zeros((size, size), dtype=np.float64)
    cell_layer[rr, cc] = cell["peak"]
    cell_layer = gaussian_filter(cell_layer, sigma=cell["blur"])

    # Internal granularity
    texture = rng.normal(0, cell["peak"] * CELL_TEXTURE_STD_FRAC, (size, size))
    texture = gaussian_filter(texture, sigma=1.8)
    cell_layer += texture * (cell_layer > cell["peak"] * 0.08)

    canvas += cell_layer


def get_cell_mask(cell, size=IMG_SIZE):
    """Binary mask of a single cell."""
    rr, cc = ellipse(
        int(cell["cy"]),
        int(cell["cx"]),
        int(cell["semi_major"]),
        int(cell["semi_minor"]),
        shape=(size, size),
        rotation=cell["angle"],
    )
    mask = np.zeros((size, size), dtype=bool)
    mask[rr, cc] = True
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Component 3: Filament skeletons
# ─────────────────────────────────────────────────────────────────────────────


def generate_filament_spine(start_y, start_x, length, rng):
    """Curved filament spine via random walk with constrained curvature."""
    theta = rng.uniform(0, 2 * np.pi)
    max_curv = rng.uniform(*FILAMENT_CURVATURE)

    points = [(start_y, start_x)]
    for _ in range(length):
        theta += rng.normal(0, max_curv)
        points.append((points[-1][0] + np.cos(theta), points[-1][1] + np.sin(theta)))
    return np.array(points)


def rasterise_spine(skeleton, spine, size):
    """Draw anti-aliased line segments along a spine."""
    for i in range(len(spine) - 1):
        y0 = int(np.clip(round(spine[i, 0]), 0, size - 1))
        x0 = int(np.clip(round(spine[i, 1]), 0, size - 1))
        y1 = int(np.clip(round(spine[i + 1, 0]), 0, size - 1))
        x1 = int(np.clip(round(spine[i + 1, 1]), 0, size - 1))
        rr, cc, val = line_aa(y0, x0, y1, x1)
        valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
        skeleton[rr[valid], cc[valid]] = np.maximum(skeleton[rr[valid], cc[valid]], val[valid])


def render_filament(canvas, mask_canvas, cell, size=IMG_SIZE, rng=None):
    """
    Render a filament inside a cell. Returns True if successfully placed.
    """
    if rng is None:
        rng = np.random.default_rng()

    cell_mask = get_cell_mask(cell, size)
    if cell_mask.sum() < 30:  # noqa: PLR2004
        return False

    # Lognormal length for heavy-tailed area distribution
    length = int(
        np.clip(rng.lognormal(FILAMENT_LENGTH_MU, FILAMENT_LENGTH_SIGMA), *FILAMENT_LENGTH_CLIP)
    )

    intensity = rng.uniform(*FILAMENT_INTENSITY)

    # Width correlates with length (longer filaments tend slightly wider)
    width_sigma = rng.uniform(*FILAMENT_WIDTH_BASE) + FILAMENT_WIDTH_LENGTH_COEF * length
    width_sigma = np.clip(width_sigma, *FILAMENT_WIDTH_CLIP)

    start_y = cell["cy"] + rng.normal(0, cell["semi_major"] * 0.25)
    start_x = cell["cx"] + rng.normal(0, cell["semi_minor"] * 0.25)

    spine = generate_filament_spine(start_y, start_x, length, rng)

    # Optional branch
    branch_spine = None
    if rng.random() < FILAMENT_BRANCH_PROB and len(spine) > 4:  # noqa: PLR2004
        bi = rng.integers(2, len(spine) - 1)
        bl = rng.integers(3, max(4, length // 3))
        branch_spine = generate_filament_spine(spine[bi, 0], spine[bi, 1], bl, rng)

    # Rasterise
    skeleton = np.zeros((size, size), dtype=np.float64)
    rasterise_spine(skeleton, spine, size)
    if branch_spine is not None:
        rasterise_spine(skeleton, branch_spine, size)

    # Gaussian cross-section for width
    filament_img = gaussian_filter(skeleton, sigma=width_sigma)
    if filament_img.max() < 1e-8:  # noqa: PLR2004
        return False
    filament_img /= filament_img.max()

    # Soft constraint to cell interior
    cell_soft = gaussian_filter(cell_mask.astype(np.float64), sigma=1.5)
    filament_img *= cell_soft

    # Binary mask
    fil_binary = (filament_img > FILAMENT_MASK_THRESHOLD) & cell_mask

    area = fil_binary.sum()
    if area < FILAMENT_MIN_AREA:
        return False

    # Add to image and mask
    canvas += filament_img * intensity
    mask_canvas |= fil_binary
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────


def generate_sample(rng=None, force_filament=None):
    """
    Generate one (image, filament_mask) pair.

    Returns: image (uint8), mask (uint8 0/255), has_filament (bool)
    """
    if rng is None:
        rng = np.random.default_rng()
    size = IMG_SIZE

    image = generate_background(size, rng)

    n_cells = rng.integers(*CELLS_PER_IMAGE)
    cells = sample_cell_params(n_cells, size, rng)
    for cell in cells:
        render_cell_body(image, cell, rng)

    filament_mask = np.zeros((size, size), dtype=bool)
    any_filament = False

    if force_filament is not False:
        # Real data: exactly 1 connected component per mask
        host_idx = rng.integers(len(cells))
        if rng.random() < FILAMENT_PROB or force_filament:  # noqa: SIM102
            if render_filament(image, filament_mask, cells[host_idx], size, rng):
                any_filament = True

        if force_filament and not any_filament:
            for _ in range(10):
                cell = cells[rng.integers(len(cells))]
                if render_filament(image, filament_mask, cell, size, rng):
                    any_filament = True
                    break

    image = np.clip(image, 0, 255).astype(np.uint8)
    mask = filament_mask.astype(np.uint8) * 255
    return image, mask, any_filament


# ─────────────────────────────────────────────────────────────────────────────
# Batch generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_dataset(output_dir, n_samples=200, filament_fraction=0.6, seed=42):
    """Generate a synthetic dataset with balanced filament/no-filament samples."""
    rng = np.random.default_rng(seed)

    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    n_with = int(n_samples * filament_fraction)
    schedule = [True] * n_with + [False] * (n_samples - n_with)
    rng.shuffle(schedule)

    stats = {"total": 0, "with_filament": 0, "mask_areas": []}

    for i, force in enumerate(schedule):
        image, mask, has_fil = generate_sample(rng, force_filament=force)

        fname = f"synthetic_{i:04d}"
        tifffile.imwrite(os.path.join(img_dir, f"{fname}.tif"), image)
        tifffile.imwrite(os.path.join(mask_dir, f"{fname}.tif"), mask)

        stats["total"] += 1
        if has_fil:
            stats["with_filament"] += 1
            stats["mask_areas"].append((mask > 0).sum())

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1:>4d}/{n_samples}] generated")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic URA7/URA8 yeast cell + filament data"
    )
    parser.add_argument("--output_dir", type=str, default="./synthetic_data")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--filament_fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.n_samples} synthetic samples...")
    print(f"  Output: {args.output_dir}")
    print(f"  Filament fraction: {args.filament_fraction:.0%}")
    print(f"  Seed: {args.seed}\n")

    stats = generate_dataset(**vars(args))

    areas = stats["mask_areas"]
    print("\nDone!")
    print(f"  Total:         {stats['total']}")
    print(f"  With filament: {stats['with_filament']}")
    print(f"  Without:       {stats['total'] - stats['with_filament']}")
    if areas:
        print(
            f"  Mask area:     mean={np.mean(areas):.1f}, "
            f"median={np.median(areas):.0f}, range=[{min(areas)}, {max(areas)}]"
        )
        print("  (Real GT:      mean=37.6, median=26, range=[7, 144])")
    print(f"\n  Images: {args.output_dir}/images/*.tif")
    print(f"  Masks:  {args.output_dir}/masks/*.tif")
