from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scipy.ndimage import gaussian_filter
from skimage import filters, io, measure, morphology, util
from skimage.filters import frangi
from skimage.morphology import remove_small_objects
from tqdm.auto import tqdm

from src.biohack.utils import read_yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FilamentConfig:
    """
    Parameters for the first-pass filament segmentation pipeline.
    
    """

    # Normalization
    clip_low_percentile: float = 1.0
    clip_high_percentile: float = 99.0

    # Denoising
    gaussian_sigma: float = 1.0

    # Percentile thresholding
    foreground_percentile: float = 92.0

    # Local thresholding (computed for diagnostics; combination may ignore it)
    local_block_size: int = 35  # must be odd
    local_offset: float = -0.01  # more negative -> more permissive

    # Filament enhancement (Frangi)
    frangi_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)
    frangi_threshold_percentile: float = 85.0

    # Cleanup
    min_object_size: int = 20

    # Presence flag
    min_pixels_for_presence: int = 20

    # Plotting
    figure_dpi: int = 140
    cmap: str = "gray"

    # Optional (also read from YAML for :func:`zero_shot_run`)
    verbose: bool = False
    input_dir: Path | str = "data/separated_frames/"
    output_dir: Path | str = "temp/"
    max_workers: int = 6
    # Human-readable run label; if None, :func:`process_directory` sets one from time + counts
    run_name: str | None = None


def filament_config_from_mapping(data: FilamentConfig | dict[str, Any]) -> FilamentConfig:
    """Build :class:`FilamentConfig` from a mapping (e.g. YAML). Ignores unknown keys."""
    if isinstance(data, FilamentConfig):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"Expected FilamentConfig or dict, got {type(data)!r}")
    field_names = {f.name for f in dataclasses.fields(FilamentConfig)}
    kwargs = {k: v for k, v in data.items() if k in field_names}
    if "frangi_sigmas" in kwargs and isinstance(kwargs["frangi_sigmas"], list):
        kwargs["frangi_sigmas"] = tuple(kwargs["frangi_sigmas"])
    for key in ("input_dir", "output_dir"):
        if key in kwargs and kwargs[key] is not None and not isinstance(kwargs[key], Path):
            kwargs[key] = Path(kwargs[key])
    return FilamentConfig(**kwargs)


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
    original_input_dir: str,
    run_dir: Path,
    raw_inputs_dir: Path,
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
            f"original_input_dir: {original_input_dir}",
            f"run_output_dir: {run_dir.resolve()}",
            f"raw_inputs_dir: {raw_inputs_dir.resolve()}",
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


def process_image(image: np.ndarray, config: FilamentConfig) -> dict[str, Any]:
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


def detect_filaments(image: np.ndarray, config: FilamentConfig | None = None) -> np.ndarray:
    """
    Isolate filament-like structures; returns a boolean (or binary) mask.

    This is a thin wrapper around :func:`process_image`.
    """
    cfg = config if config is not None else FilamentConfig()
    return process_image(image, cfg)["final_mask"]


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
    config: FilamentConfig,
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
        save_binary_mask(results["final_mask"], out / "masks" / f"{path.stem}_mask_{t}_{channel}.png")
        plot_pipeline_results(
                results,
                title=f"{path.stem}_{t}_{channel}",
                save_path=out / "images" / f"{path.stem}_pipeline_{t}_{channel}.png",
                dpi=config.figure_dpi,
                cmap=config.cmap,
            )
        results_dict[t] = results
    return results_dict

def process_single_image_file(
    image_path: Path | str,
    output_dir: Path | str,
    config: FilamentConfig,
) -> dict[str, Any]:
    """
    Run the pipeline on one file; write mask under ``output_dir/masks/`` and
    diagnostic figure under ``output_dir/images/``.
    """
    path = Path(image_path)
    out = Path(output_dir)
    image = load_image(path)
    results = process_image(image, config=config)

    stem = path.stem
    mask_path = out / "masks" / f"{stem}_mask.png"
    fig_path = out / "images" / f"{stem}_pipeline.png"

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

    ``payload`` is ``(image_path_str, output_dir_str, config_dict)`` with
    ``config_dict`` from :func:`dataclasses.asdict` on :class:`FilamentConfig`.
    """
    image_path_str, output_dir_str, config_dict = payload
    cfg = FilamentConfig(**config_dict)
    path = Path(image_path_str)
    results = process_single_image_file(path, Path(output_dir_str), cfg)
    return path.name, results


def process_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    config: FilamentConfig | dict[str, Any],
    suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    *,
    max_workers: int = 4,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run the pipeline on all matching images under ``input_dir``.

    For each run:

    - A unique ``run_uid`` is generated.
    - A per-run folder ``output_dir/<run_uid>/`` is created containing:
      ``raw_inputs/`` (inputs **copied** here), ``masks/``, ``images/``, and
      ``metadata.txt`` at the run root.

    Returns a dict with ``run_uid``, ``run_name``, paths, ``metadata_path``, and
    ``results`` (mapping filename -> per-image pipeline dict).
    """

    cfg = filament_config_from_mapping(config)

    if verbose:
        print("-" * 50)
        print("-" * 20, " Filament detection run ", "-" * 20)
        print("-" * 50)
        print(f"Processing directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print("-" * 50)
        print(f"Config: {cfg}")
        print("-" * 50)
        print(f"Max workers: {max_workers}")
        print(f"Verbose: {verbose}")
        print("-" * 50)

    in_dir = Path(input_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes
    )

    if not image_paths:
        return {
            "run_uid": None,
            "run_name": None,
            "run_dir": None,
            "raw_inputs_dir": None,
            "metadata_path": None,
            "results": {},
        }

    run_uid = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    started_at_str = started_at.isoformat()

    run_dir = out_root / run_uid
    raw_run_dir = run_dir / "raw_inputs"
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    raw_run_dir.mkdir(parents=True, exist_ok=True)

    original_input_dir_str = str(in_dir)

    copied_paths: list[Path] = []
    for p in image_paths:
        dest = raw_run_dir / p.name
        if dest.exists():
            raise FileExistsError(f"Destination already exists (name clash): {dest}")
        shutil.copy2(str(p.resolve()), str(dest))
        copied_paths.append(dest)

    run_dir_str = str(run_dir.resolve())
    config_dict = asdict(cfg)

    all_results: dict[str, dict[str, Any]] = {}

    if max_workers <= 1:
        for image_path in tqdm(
            copied_paths,
            desc="Processing images",
            unit="img",
        ):
            all_results[image_path.name] = process_single_image_file(
                image_path, run_dir, cfg
            )
    else:
        workers = max(1, min(max_workers, len(copied_paths)))
        payloads = [(str(p.resolve()), run_dir_str, config_dict) for p in copied_paths]

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
        if isinstance(v, Path):
            pipeline_meta[k] = str(v)
        elif isinstance(v, tuple):
            pipeline_meta[k] = list(v)

    write_run_metadata(
        meta_path,
        run_uid=run_uid,
        run_name=run_name,
        started_at_utc=started_at_str,
        finished_at_utc=finished_at_str,
        n_images=n_images,
        n_images_with_filament=n_with_fil,
        total_connected_components=total_components,
        original_input_dir=original_input_dir_str,
        run_dir=run_dir,
        raw_inputs_dir=raw_run_dir,
        pipeline_config=pipeline_meta,
    )

    return {
        "run_uid": run_uid,
        "run_name": run_name,
        "run_dir": run_dir,
        "raw_inputs_dir": raw_run_dir,
        "metadata_path": meta_path,
        "results": all_results,
    }


def zero_shot_run():
    """
    Run entire extraction pipeline without having to set any parameters or paths.
    Any parameters or paths will be read from the config file.
    """

    try:
        raw_config = read_yaml("config/image_detection.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Config file not found. Please create a config file at config/image_detection.yaml."
        ) from None

    input_dir = raw_config.get("input_dir", "data/separated_frames/")
    output_dir = raw_config.get("output_dir", "temp/")
    max_workers = int(raw_config.get("max_workers", 6))
    verbose = bool(raw_config.get("verbose", False))
    cfg = filament_config_from_mapping(raw_config)

    batch = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        config=cfg,
        max_workers=max_workers,
        verbose=verbose,
    )

    print("Segmentation completed successfully.")
    print(f"run_uid: {batch['run_uid']}")
    print(f"run_name: {batch['run_name']}")
    print(f"Run directory: {batch['run_dir']}")
    print(f"Raw inputs: {batch['raw_inputs_dir']}")
    print(f"Metadata: {batch['metadata_path']}")
