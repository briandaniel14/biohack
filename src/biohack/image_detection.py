import json
import multiprocessing as mp
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scipy.ndimage import gaussian_filter
from skimage import filters, io, measure, morphology, util
from skimage.filters import frangi
from skimage.morphology import remove_small_objects
from tqdm.auto import tqdm

from src.biohack.experiment_config import ExperimentConfig
from src.biohack.utils import read_yaml

from src.biohack.constants import (
    DATASET_SUBDIR_BRIGHTFIELD,
    DATASET_SUBDIR_GFP,
    RUN_SUBDIR_FILAMENT_MASK,
    RUN_SUBDIR_DIAGNOSTICS,
)

# Dataset layout under ``dataset_directory``; run artifacts under ``results_directory / <run_uid> /``.

def default_run_name(
    started_at: datetime,
    n_images: int,
    n_images_with_filament: int,
) -> str:
    """
    Default run label: UTC timestamp, image count, count of images with filament present.
    """
    ts = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{n_images}img_{n_images_with_filament}fil"


def write_run_metadata(
    path: Path,
    *,
    run_uid: str,
    run_name: str,
    started_at_utc: str,
    finished_at_utc: str,
    n_images: int,
    n_images_with_filament: int,
    total_connected_components: int,
    dataset_source_directory: str,
    run_dir: Path,
    gfp_working_directory: Path,
    brightfield_snapshot_directory: Path,
    filament_mask_directory: Path,
    diagnostics_directory: Path,
    pipeline_config: dict[str, Any],
) -> None:
    """Write a plain-text metadata file for one batch run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_json = json.dumps(pipeline_config, indent=2, default=str)
    body = "\n".join(
        [
            f"run_uid: {run_uid}",
            f"run_name: {run_name}",
            f"started_at_utc: {started_at_utc}",
            f"finished_at_utc: {finished_at_utc}",
            f"images_processed: {n_images}",
            f"images_with_filament_present: {n_images_with_filament}",
            f"total_connected_components (sum over images): {total_connected_components}",
            f"dataset_source_directory: {dataset_source_directory}",
            f"run_output_dir: {run_dir.resolve()}",
            f"gfp_working_directory (snapshot): {gfp_working_directory.resolve()}",
            f"brightfield_snapshot_directory: {brightfield_snapshot_directory.resolve()}",
            f"filament_mask_directory: {filament_mask_directory.resolve()}",
            f"diagnostics_directory: {diagnostics_directory.resolve()}",
            "",
            "--- pipeline_config (JSON) ---",
            cfg_json,
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def ensure_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Convert image to float grayscale in [0, 1] if needed."""
    if image.ndim == 3:
        image = util.img_as_float(image)
        image = image[..., :3].mean(axis=-1)
    else:
        image = util.img_as_float(image)
    return image


def robust_normalize(image: np.ndarray, low_p: float, high_p: float) -> np.ndarray:
    """Clip to robust percentiles and rescale to [0, 1]."""
    lo = np.percentile(image, low_p)
    hi = np.percentile(image, high_p)

    if hi <= lo:
        return np.clip(image, 0.0, 1.0)

    clipped = np.clip(image, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


def denoise_image(image: np.ndarray, sigma: float) -> np.ndarray:
    """Light Gaussian smoothing."""
    if sigma <= 0:
        return image.copy()
    return gaussian_filter(image, sigma=sigma)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def percentile_threshold_mask(image: np.ndarray, percentile: float) -> np.ndarray:
    """Keep pixels >= percentile threshold."""
    threshold = np.percentile(image, percentile)
    return image >= threshold


def local_threshold_mask(image: np.ndarray, block_size: int, offset: float) -> np.ndarray:
    """Adaptive / local thresholding."""
    if block_size % 2 == 0:
        raise ValueError("local_block_size must be odd.")

    local_thresh = filters.threshold_local(image, block_size=block_size, offset=offset)
    return image > local_thresh


def enhance_filaments_frangi(image: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    """
    Frangi filter for line- / ridge-like structures.

    ``black_ridges=False``: filaments are bright on dark background.
    """
    enhanced = frangi(image, sigmas=sigmas, black_ridges=False)
    enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)

    if enhanced.max() > enhanced.min():
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
    else:
        enhanced = np.zeros_like(enhanced)

    return enhanced


def threshold_enhancement_map(enhanced: np.ndarray, percentile: float) -> np.ndarray:
    """Threshold filamentness map by percentile."""
    threshold = np.percentile(enhanced, percentile)
    return enhanced >= threshold


def combine_masks(
    percentile_mask: np.ndarray,
    local_mask: np.ndarray,
    enhancement_mask: np.ndarray,
) -> np.ndarray:
    """
    Combine masks.

    Default: ``percentile_mask & enhancement_mask``.
    (Alternative: ``enhancement_mask & (percentile_mask | local_mask)``.)
    """
    _ = local_mask  # kept for API compatibility / future use
    return percentile_mask & enhancement_mask


def cleanup_mask(mask: np.ndarray, min_object_size: int) -> np.ndarray:
    """Remove tiny connected components."""
    return remove_small_objects(mask, min_size=min_object_size)


def derive_filament_present_flag(final_mask: np.ndarray, min_pixels_for_presence: int) -> bool:
    """Return True if foreground has at least ``min_pixels_for_presence`` pixels."""
    return int(final_mask.sum()) >= min_pixels_for_presence


def compute_summary_stats(mask: np.ndarray) -> dict[str, Any]:
    """Foreground pixel count and number of connected components."""
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)

    return {
        "foreground_pixels": int(mask.sum()),
        "num_components": len(props),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def process_image(image: np.ndarray, config: ExperimentConfig) -> dict[str, Any]:
    """
    Run the full first-pass filament pipeline on one image.

    Returns intermediate arrays, ``final_mask``, ``filament_present``, stats, and
    serialized ``config``.
    """
    raw = ensure_grayscale_float(image)

    normalized = robust_normalize(
        raw,
        low_p=config.clip_low_percentile,
        high_p=config.clip_high_percentile,
    )

    denoised = denoise_image(normalized, sigma=config.gaussian_sigma)

    pct_mask = percentile_threshold_mask(
        normalized,
        percentile=config.foreground_percentile,
    )

    loc_mask = local_threshold_mask(
        denoised,
        block_size=config.local_block_size,
        offset=config.local_offset,
    )

    enhanced = enhance_filaments_frangi(denoised, sigmas=config.frangi_sigmas)

    enh_mask = threshold_enhancement_map(
        enhanced,
        percentile=config.frangi_threshold_percentile,
    )

    combined_mask = combine_masks(
        percentile_mask=pct_mask,
        local_mask=loc_mask,
        enhancement_mask=enh_mask,
    )

    final_mask = cleanup_mask(combined_mask, min_object_size=config.min_object_size)

    filament_present = derive_filament_present_flag(
        final_mask,
        min_pixels_for_presence=config.min_pixels_for_presence,
    )

    stats = compute_summary_stats(final_mask)
    stats["filament_present"] = filament_present

    return {
        "raw": raw,
        "normalized": normalized,
        "denoised": denoised,
        "percentile_mask": pct_mask,
        "local_mask": loc_mask,
        "enhanced": enhanced,
        "enhancement_mask": enh_mask,
        "combined_mask": combined_mask,
        "final_mask": final_mask,
        "filament_present": filament_present,
        "stats": stats,
        "config": asdict(config),
    }


def detect_filaments(image: np.ndarray, config: ExperimentConfig | None = None) -> np.ndarray:
    """
    Isolate filament-like structures; returns a boolean (or binary) mask.

    This is a thin wrapper around :func:`process_image`.
    Dataset paths on ``config`` are ignored when only the image array is processed.
    """
    return process_image(image, config)["final_mask"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pipeline_results(
    results: dict[str, Any],
    title: str | None = None,
    save_path: Path | None = None,
    dpi: int = 140,
    cmap: str = "gray",
) -> None:
    """Multi-panel debug figure for all major pipeline steps."""
    import matplotlib.pyplot as plt

    raw = results["raw"]
    normalized = results["normalized"]
    denoised = results["denoised"]
    percentile_mask = results["percentile_mask"]
    local_mask = results["local_mask"]
    enhanced = results["enhanced"]
    enhancement_mask = results["enhancement_mask"]
    combined_mask = results["combined_mask"]
    final_mask = results["final_mask"]
    filament_present = results["filament_present"]
    stats = results["stats"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=dpi)
    axes = axes.ravel()

    panels: list[tuple[str, np.ndarray | None]] = [
        ("Raw image", raw),
        ("Normalized", normalized),
        ("Denoised", denoised),
        ("Percentile mask", percentile_mask),
        ("Local mask (unused)", local_mask),
        ("Frangi enhanced", enhanced),
        ("Enhancement mask", enhancement_mask),
        ("Combined mask", combined_mask),
        ("Final mask", final_mask),
        ("Overlay: final on raw", raw),
        ("Skeleton (optional preview)", morphology.skeletonize(final_mask)),
        ("Summary", None),
    ]

    for ax, (panel_title, panel_image) in zip(axes, panels, strict=True):
        ax.set_title(panel_title)
        ax.axis("off")

        if panel_title == "Overlay: final on raw":
            ax.imshow(raw, cmap=cmap)
            masked = np.ma.masked_where(~final_mask, final_mask)
            ax.imshow(masked, alpha=0.8)
        elif panel_title == "Summary":
            summary_text = (
                f"Filament present: {filament_present}\n"
                f"Foreground pixels: {stats['foreground_pixels']}\n"
                f"Connected components: {stats['num_components']}"
            )
            ax.text(
                0.05,
                0.95,
                summary_text,
                va="top",
                ha="left",
                fontsize=12,
                family="monospace",
            )
        else:
            ax.imshow(panel_image, cmap=cmap)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_image(image_path: Path | str, channel: int = None, timepoint: int = None) -> np.ndarray:
    """Load image from disk."""
    image = io.imread(str(image_path))
    if image.ndim == 4 and channel is not None and timepoint is not None:
        image = image[timepoint,channel,:,:]
    return image


def save_binary_mask(mask: np.ndarray, output_path: Path | str) -> None:
    """Save binary mask as uint8 image (0 or 255)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(path), (mask.astype(np.uint8) * 255), check_contrast=False)


def process_time_series_image(
    image_path: Path | str,
    output_dir: Path | str,
    config: ExperimentConfig,
    channel: int = 1,
) -> dict[str, Any]:
    """
    Run the pipeline on a time series image.
    """
    path = Path(image_path)
    out = Path(output_dir)
    base_image = load_image(path)
    temporal_length = base_image.shape[0]
    print(temporal_length)
    results_dict = {}
    for t in range(temporal_length):
        print(t)
        image = load_image(path, channel=channel, timepoint=t)
        print(image.shape)
        results = process_image(image, config=config)
        save_binary_mask(
            results["final_mask"],
            out / RUN_SUBDIR_FILAMENT_MASK / f"{path.stem}_mask_{t}_{channel}.png",
        )
        plot_pipeline_results(
                results,
                title=f"{path.stem}_{t}_{channel}",
                save_path=out / RUN_SUBDIR_DIAGNOSTICS / f"{path.stem}_pipeline_{t}_{channel}.png",
                dpi=config.figure_dpi,
                cmap=config.cmap,
            )
        results_dict[t] = results
    return results_dict

def process_single_image_file(
    image_path: Path | str,
    run_dir: Path | str,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """
    Run the pipeline on one file; write mask under ``run_dir/filament_mask/`` and
    diagnostic figure under ``run_dir/diagnostics/``.
    """
    path = Path(image_path)
    out = Path(run_dir)
    image = load_image(path)
    results = process_image(image, config=config)

    stem = path.stem
    mask_path = out / RUN_SUBDIR_FILAMENT_MASK / f"{stem}_mask.png"
    fig_path = out / RUN_SUBDIR_DIAGNOSTICS / f"{stem}_pipeline.png"

    save_binary_mask(results["final_mask"], mask_path)

    plot_pipeline_results(
        results,
        title=stem,
        save_path=fig_path,
        dpi=config.figure_dpi,
        cmap=config.cmap,
    )

    return results


def _process_directory_worker(
    payload: tuple[str, str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """
    Picklable worker for :func:`process_directory`.

    ``payload`` is ``(image_path_str, run_dir_str, config_dict)`` with
    ``run_dir_str`` the per-run artifact root (``filament_mask/``, ``diagnostics/``).
    """
    image_path_str, output_dir_str, config_dict = payload
    cfg = ExperimentConfig(**config_dict)
    path = Path(image_path_str)
    results = process_single_image_file(path, Path(output_dir_str), cfg)
    return path.name, results


def process_directory(
    experiment: ExperimentConfig,
    suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    *,
    brightfield_subdir: str = DATASET_SUBDIR_BRIGHTFIELD,
    gfp_subdir: str = DATASET_SUBDIR_GFP,
    max_workers: int | None = None,
    verbose: bool | None = None,
    run_uid: str | None = None,
) -> dict[str, Any]:
    """
    Run filament detection using :class:`ExperimentConfig` paths and parameters.

    - On a **new** run (no existing ``results_directory/<run_uid>/``): copies
      ``brightfield_subdir`` and ``gfp_subdir`` from ``dataset_directory`` into
      the run folder. That snapshot is never overwritten on later calls.
    - On **reuse** (same ``run_uid`` and the run folder already contains the BF/GFP
      snapshot): skips copying and only refreshes segmentation outputs under
      ``filament_mask/`` and ``diagnostics/``. Other subfolders (e.g.
      ``statistics/``, ``cellpose_mask/``) are left unchanged.
    - If ``run_uid`` is omitted, a new UUID is used (always a fresh snapshot copy).

    Images are always read from the run’s ``gfp`` snapshot. ``metadata.txt`` is
    rewritten after each segmentation pass.

    ``max_workers`` and ``verbose`` default to values on ``experiment`` when omitted.
    """

    cfg = experiment
    mw = cfg.max_workers if max_workers is None else max_workers
    vb = cfg.verbose if verbose is None else verbose

    if vb:
        print("-" * 50)
        print("-" * 20, " Filament detection run ", "-" * 20)
        print("-" * 50)
        print(f"Dataset directory: {cfg.dataset_directory}")
        print(f"Results directory: {cfg.results_directory}")
        print("-" * 50)
        print(f"Config: {cfg}")
        print("-" * 50)
        print(f"Max workers: {mw}")
        print(f"Verbose: {vb}")
        print("-" * 50)

    dataset_root = Path(cfg.dataset_directory).resolve()
    results_root = Path(cfg.results_directory).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    src_brightfield = dataset_root / brightfield_subdir
    src_gfp = dataset_root / gfp_subdir

    if run_uid is not None:
        run_uid_final = run_uid
    else:
        run_uid_final = str(uuid.uuid4())

    run_dir = results_root / run_uid_final
    reuse_snapshot = False

    if run_dir.exists():
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Run path exists but is not a directory: {run_dir}")
        dst_brightfield = run_dir / brightfield_subdir
        dst_gfp = run_dir / gfp_subdir
        if not dst_brightfield.is_dir() or not dst_gfp.is_dir():
            raise FileNotFoundError(
                f"Run directory {run_dir} exists but is missing a brightfield or gfp "
                "snapshot subdirectory; refusing to copy over it."
            )
        reuse_snapshot = True
    else:
        if not src_brightfield.is_dir():
            raise FileNotFoundError(
                f"Expected brightfield data at {src_brightfield}"
            )
        if not src_gfp.is_dir():
            raise FileNotFoundError(f"Expected GFP data at {src_gfp}")

        src_image_paths = sorted(
            p
            for p in src_gfp.iterdir()
            if p.is_file() and p.suffix.lower() in suffixes
        )

        if not src_image_paths:
            return {
                "run_uid": None,
                "run_name": None,
                "run_dir": None,
                "dataset_source_directory": str(dataset_root),
                "gfp_dir": None,
                "brightfield_dir": None,
                "filament_mask_dir": None,
                "diagnostics_dir": None,
                "metadata_path": None,
                "results": {},
                "segmentation_snapshot_reused": False,
            }

        run_dir.mkdir(parents=True, exist_ok=False)
        dst_brightfield = run_dir / brightfield_subdir
        dst_gfp = run_dir / gfp_subdir
        shutil.copytree(src_brightfield, dst_brightfield)
        shutil.copytree(src_gfp, dst_gfp)

    started_at = datetime.now(timezone.utc)
    started_at_str = started_at.isoformat()

    filament_mask_dir = run_dir / RUN_SUBDIR_FILAMENT_MASK
    diagnostics_dir = run_dir / RUN_SUBDIR_DIAGNOSTICS
    if reuse_snapshot:
        if filament_mask_dir.is_dir():
            shutil.rmtree(filament_mask_dir)
        if diagnostics_dir.is_dir():
            shutil.rmtree(diagnostics_dir)
    filament_mask_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in dst_gfp.iterdir() if p.is_file() and p.suffix.lower() in suffixes
    )

    if not image_paths:
        raise RuntimeError(
            f"No images matching {suffixes!r} under GFP snapshot {dst_gfp}; "
            "cannot run segmentation."
        )

    original_dataset_str = str(dataset_root)

    run_dir_str = str(run_dir.resolve())
    config_dict = asdict(cfg)

    all_results: dict[str, dict[str, Any]] = {}

    if mw <= 1:
        for image_path in tqdm(
            image_paths,
            desc="Processing images",
            unit="img",
        ):
            all_results[image_path.name] = process_single_image_file(
                image_path, run_dir, cfg
            )
    else:
        workers = max(1, min(mw, len(image_paths)))
        payloads = [(str(p.resolve()), run_dir_str, config_dict) for p in image_paths]

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(_process_directory_worker, payload): payload[0]
                for payload in payloads
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing images",
                unit="img",
            ):
                path_str = futures[fut]
                name = Path(path_str).name
                try:
                    result_name, results = fut.result()
                    all_results[result_name] = results
                except Exception as e:
                    tqdm.write(f"Failed: {name} ({e})")
                    raise

    finished_at = datetime.now(timezone.utc)
    finished_at_str = finished_at.isoformat()

    n_images = len(all_results)
    n_with_fil = sum(1 for r in all_results.values() if r.get("filament_present"))
    total_components = sum(
        int(r.get("stats", {}).get("num_components", 0)) for r in all_results.values()
    )

    run_name = cfg.run_name
    if not run_name or not str(run_name).strip():
        run_name = default_run_name(started_at, n_images, n_with_fil)

    meta_path = run_dir / "metadata.txt"
    pipeline_meta = asdict(cfg)
    for k, v in list(pipeline_meta.items()):
        if isinstance(v, tuple):
            pipeline_meta[k] = list(v)

    write_run_metadata(
        meta_path,
        run_uid=run_uid_final,
        run_name=run_name,
        started_at_utc=started_at_str,
        finished_at_utc=finished_at_str,
        n_images=n_images,
        n_images_with_filament=n_with_fil,
        total_connected_components=total_components,
        dataset_source_directory=original_dataset_str,
        run_dir=run_dir,
        gfp_working_directory=dst_gfp,
        brightfield_snapshot_directory=dst_brightfield,
        filament_mask_directory=filament_mask_dir,
        diagnostics_directory=diagnostics_dir,
        pipeline_config=pipeline_meta,
    )

    print(f"Run directory: {run_dir}")
    print(f"Dataset source directory: {dataset_root}")
    print(f"GFP directory: {dst_gfp}")
    print(f"Brightfield directory: {dst_brightfield}")
    print(f"Filament mask directory: {filament_mask_dir}")
    print(f"Diagnostics directory: {diagnostics_dir}")
    print(f"Metadata path: {meta_path}")
    print(f"Results: {all_results}")

    return {
        "run_uid": run_uid_final,
        "run_name": run_name,
        "run_dir": run_dir,
        "dataset_source_directory": dataset_root,
        "gfp_dir": dst_gfp,
        "brightfield_dir": dst_brightfield,
        "filament_mask_dir": filament_mask_dir,
        "diagnostics_dir": diagnostics_dir,
        "metadata_path": meta_path,
        "results": all_results,
        "segmentation_snapshot_reused": reuse_snapshot,
    }

