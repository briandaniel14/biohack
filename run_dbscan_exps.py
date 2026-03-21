"""Run DBSCAN grid search over images in parallel (max 6 worker processes)."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

from src.biohack.utils import (
    load_grayscale_image,
    load_image_paths,
    set_lowest_percent_to_black,
)

# One copy per worker (set by pool initializer).
_WORKER_SAMPLES: list[ImageSample] | None = None

MAX_WORKERS = 6
INTENSITY_WEIGHT = 0.3
BLACK_PERCENT = 99.5
RNG_SEED = 42

# Hyperparameter grids (edit here)
EPS_SEARCH_SPACE = [2.5, 3, 3.5]
MIN_SAMPLES_SEARCH_SPACE = [5, 8]


@dataclass
class ImageSample:
    path: Path
    original_image: np.ndarray
    blacked_image: np.ndarray


def _init_worker(samples: list[ImageSample]) -> None:
    global _WORKER_SAMPLES
    _WORKER_SAMPLES = samples


def _run_single_experiment(task: tuple[float, int, int]) -> Path:
    """One job: DBSCAN + triptych for (eps, min_samples, image_index)."""
    eps, min_samples, idx = task
    if _WORKER_SAMPLES is None:
        raise RuntimeError("Pool initializer did not set _WORKER_SAMPLES")

    sample = _WORKER_SAMPLES[idx]
    cleaned = sample.blacked_image

    mask = cleaned > 0
    rows, cols = np.where(mask)

    if rows.size == 0:
        labels = np.array([], dtype=np.int32)
        n_clusters = 0
        cluster_rgb = np.zeros((*cleaned.shape, 3), dtype=np.uint8)
    else:
        x = cols.astype(np.float32)
        y = rows.astype(np.float32)
        intensity = cleaned[rows, cols].astype(np.float32)
        intensity = (intensity / 255.0) * INTENSITY_WEIGHT
        X = np.column_stack([x, y, intensity])

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        cluster_map = np.full(cleaned.shape, -1, dtype=np.int32)
        cluster_map[rows, cols] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        cluster_rgb = np.zeros((*cluster_map.shape, 3), dtype=np.uint8)
        valid = labels >= 0
        unique_labels = np.unique(labels[valid])
        rng = np.random.default_rng(RNG_SEED)
        colors = rng.integers(40, 255, size=(len(unique_labels), 3), dtype=np.uint8)
        for color_idx, label in enumerate(unique_labels):
            pix = labels == label
            cluster_rgb[rows[pix], cols[pix]] = colors[color_idx]

    output_dir = Path(f"outputs/dbscan_eps{eps}_min_samples{min_samples}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(sample.original_image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(sample.blacked_image, cmap="gray")
    axes[1].set_title("Blackened")
    axes[1].axis("off")

    axes[2].imshow(cluster_rgb)
    axes[2].set_title(f"Clustered ({n_clusters} clusters)")
    axes[2].axis("off")

    fig.suptitle(sample.path.name)
    fig.tight_layout()

    save_path = output_dir / f"{sample.path.stem}_overview.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path


def build_samples(paths: list[Path], black_percent: float) -> list[ImageSample]:
    out: list[ImageSample] = []
    for i, p in enumerate(paths):
        print(f"Image {i}, shape: {p.name}")
        img = load_grayscale_image(p)
        blacked = set_lowest_percent_to_black(img, black_percent=black_percent)
        out.append(ImageSample(p, img, blacked))
    return out


def main() -> None:
    paths = load_image_paths(r"data/separated_frames/")
    filtered_imgs = build_samples(paths, black_percent=BLACK_PERCENT)

    tasks: list[tuple[float, int, int]] = []
    for eps in EPS_SEARCH_SPACE:
        for min_samples in MIN_SAMPLES_SEARCH_SPACE:
            for i in range(len(filtered_imgs)):
                tasks.append((eps, min_samples, i))

    n_workers = min(MAX_WORKERS, max(1, mp.cpu_count() or 1))

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(filtered_imgs,),
    ) as pool:
        for _ in tqdm(
            pool.imap_unordered(_run_single_experiment, tasks, chunksize=1),
            total=len(tasks),
            desc="DBSCAN experiments",
        ):
            pass


if __name__ == "__main__":
    main()
