from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from src.biohack.utils import read_yaml
import optuna
import numpy as np
from optuna.samplers import TPESampler

# Import your existing pipeline code here
# Adjust this import path to your project structure.
from src.biohack.image_detection import (
    FilamentConfig,
    ensure_grayscale_float,
    robust_normalize,
    denoise_image,
    percentile_threshold_mask,
    local_threshold_mask,
    enhance_filaments_frangi,
    threshold_enhancement_map,
    combine_masks,
    cleanup_mask,
    derive_filament_present_flag,
    load_image,
)

# ============================================================
# EDITABLE SEARCH SPACES
# ============================================================


# def index_masks(mask_dir: Path, mask_suffix: str) -> dict[str, Path]:
#     """
#     Build mapping:
#         image_stem -> mask_path

#     Example:
#         ch20_..._frame_10_mask.tif
#     becomes:
#         ch20_..._frame_10 -> path
#     """
#     mask_map: dict[str, Path] = {}

#     for p in mask_dir.iterdir():
#         if not p.is_file():
#             continue

#         if not p.name.endswith(mask_suffix):
#             continue

#         stem = p.name[: -len(mask_suffix)]
#         mask_map[stem] = p

#     return mask_map

RAW_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "gaussian_sigma": {
        "type": "float",
        "low": 1.8,
        "high": 3.0,
        "step": 0.2,
    },
    "foreground_percentile": {
        "type": "float",
        "low": 98.0,
        "high": 99.99,
        "step": 0.25,
    },
    "frangi_threshold_percentile": {
        "type": "float",
        "low": 98.0,
        "high": 99.99,
        "step": 0.5,
    },
}

CLEANUP_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "min_object_size": {
        "type": "int",
        "low": 16,
        "high": 50,
        "step": 4,
    },
}

PRESENCE_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "min_pixels_for_presence": {
        "type": "int",
        "low": 15,
        "high": 50,
        "step": 4,
    },
}

# Which metric to optimize in each study
RAW_OBJECTIVE_METRIC = "f0_5"
CLEANUP_OBJECTIVE_METRIC = "f0_5"
PRESENCE_OBJECTIVE_METRIC = "f0_5"

# Default fixed config values used unless tuned in a given study
BASE_CONFIG = FilamentConfig(**read_yaml("config/image_detection.yaml"))

# ============================================================
# DATASET LOADING
# ============================================================


from pathlib import Path
from typing import Any
import numpy as np


def get_episode_key_from_stem(stem: str) -> str:
    """
    Example:
      ch20_URA7_URA8_001-crop1_frame_10 -> ch20_URA7_URA8_001-crop1
    """
    if "_frame" not in stem:
        raise ValueError(f"Filename stem does not contain '_frame': {stem}")
    return stem.split("_frame", 1)[0]


def index_masks(mask_dir: Path, mask_suffix: str) -> tuple[dict[str, Path], set[str]]:
    """
    Returns:
      mask_map: image_stem -> mask_path
      labeled_episodes: set of episode keys represented in mask_dir

    Example mask file:
      ch20_URA7_URA8_001-crop1_frame_10_mask.tif
    maps to image stem:
      ch20_URA7_URA8_001-crop1_frame_10
    """
    mask_map: dict[str, Path] = {}
    labeled_episodes: set[str] = set()

    for p in mask_dir.iterdir():
        if not p.is_file():
            continue
        if not p.name.endswith(mask_suffix):
            continue

        image_stem = p.name[: -len(mask_suffix)]  # remove "_mask.tif"
        episode_key = get_episode_key_from_stem(image_stem)

        mask_map[image_stem] = p
        labeled_episodes.add(episode_key)

    return mask_map, labeled_episodes


def load_binary_mask(mask_path: Path | str) -> np.ndarray:
    mask = load_image(mask_path)
    if mask.ndim == 3:
        mask = mask[0,:,:]
    return mask > 0


def load_dataset_from_dirs(
    image_dir: Path | str,
    mask_dir: Path | str,
    image_suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    mask_suffix: str = "_mask.tif",
) -> list[dict[str, Any]]:
    """
    Build dataset using this rule:

    - Only images from episodes that appear in mask_dir are included.
    - Within those episodes:
        * exact frame mask exists -> positive
        * no exact frame mask exists -> negative

    This prevents unrelated unlabeled episodes in image_dir from being treated as negatives.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    mask_map, labeled_episodes = index_masks(mask_dir, mask_suffix)

    dataset: list[dict[str, Any]] = []

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_suffixes
    )
    print(len(image_paths))

    n_total_images = 0
    n_used_images = 0
    n_skipped_unlabeled_episode = 0
    n_positive = 0
    n_negative = 0

    for image_path in image_paths:
        n_total_images += 1

        stem = image_path.stem
        episode_key = get_episode_key_from_stem(stem)

        # Only use images from episodes that are represented in the mask folder
        if episode_key not in labeled_episodes:
            n_skipped_unlabeled_episode += 1
            continue

        image = load_image(image_path)
        image_gray = ensure_grayscale_float(image)

        if stem in mask_map:
            gt_mask = load_binary_mask(mask_map[stem])

            if gt_mask.shape != image_gray.shape:
                raise ValueError(
                    f"Shape mismatch for {stem}: "
                    f"image={image_gray.shape}, mask={gt_mask.shape}"
                )

            filament_present = True
            n_positive += 1
        else:
            gt_mask = np.zeros(image_gray.shape, dtype=bool)
            filament_present = False
            n_negative += 1

        dataset.append(
            {
                "name": stem,
                "episode_key": episode_key,
                "image": image,
                "mask": gt_mask,
                "filament_present": filament_present,
            }
        )
        n_used_images += 1

    print("Dataset summary:")
    print(f"  Total images found: {n_total_images}")
    print(f"  Images used from labeled episodes: {n_used_images}")
    print(f"  Skipped images from unlabeled episodes: {n_skipped_unlabeled_episode}")
    print(f"  Positive frames (mask exists): {n_positive}")
    print(f"  Negative frames (no mask in labeled episode): {n_negative}")
    print(f"  Labeled episodes: {len(labeled_episodes)}")

    return dataset


# ============================================================
# PIPELINE HELPERS
# ============================================================


def build_raw_segmentation_mask(
    image: np.ndarray, config: FilamentConfig
) -> np.ndarray:
    """
    Returns combined_mask from your current pipeline.
    This is the raw segmentation stage before cleanup.
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

    # Computed for compatibility with your current pipeline, but not used in combine_masks
    loc_mask = local_threshold_mask(
        denoised,
        block_size=config.local_block_size,
        offset=config.local_offset,
    )

    enhanced = enhance_filaments_frangi(
        denoised,
        sigmas=config.frangi_sigmas,
    )

    enh_mask = threshold_enhancement_map(
        enhanced,
        percentile=config.frangi_threshold_percentile,
    )

    combined_mask = combine_masks(
        percentile_mask=pct_mask,
        local_mask=loc_mask,
        enhancement_mask=enh_mask,
    )

    return combined_mask


def build_cleaned_mask(image: np.ndarray, config: FilamentConfig) -> np.ndarray:
    raw_mask = build_raw_segmentation_mask(image, config)
    return cleanup_mask(raw_mask, min_object_size=config.min_object_size)


def derive_presence_from_mask(mask: np.ndarray, min_pixels_for_presence: int) -> bool:
    return derive_filament_present_flag(
        mask, min_pixels_for_presence=min_pixels_for_presence
    )


# ============================================================
# METRICS
# ============================================================


def compute_pixel_metrics(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, float]:
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()

    # Special empty-empty case: treat as perfect
    if gt.sum() == 0 and pred.sum() == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "f0_5": 1.0,
            "dice": 1.0,
            "iou": 1.0,
            "accuracy": 1.0,
        }

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    beta2 = beta**2
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f0_5": float(fbeta),
        "dice": float(dice),
        "iou": float(iou),
        "accuracy": float(accuracy),
    }


def compute_image_level_metrics(
    y_true: list[bool],
    y_pred: list[bool],
    beta: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)

    tp = np.logical_and(yt, yp).sum()
    fp = np.logical_and(~yt, yp).sum()
    fn = np.logical_and(yt, ~yp).sum()
    tn = np.logical_and(~yt, ~yp).sum()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    balanced_accuracy = 0.5 * (recall + specificity)

    beta2 = beta**2
    f0_5 = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "f0_5": float(f0_5),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ============================================================
# OPTUNA SEARCH SPACE HELPER
# ============================================================


def suggest_from_space(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = spec["type"]

    if kind == "float":
        return trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step"),
            log=spec.get("log", False),
        )

    if kind == "int":
        return trial.suggest_int(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step", 1),
            log=spec.get("log", False),
        )

    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])

    raise ValueError(f"Unsupported search space type for {name}: {kind}")


# ============================================================
# EVALUATION HELPERS
# ============================================================


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = metric_dicts[0].keys()
    out: dict[str, float] = {}

    for k in keys:
        values = [m[k] for m in metric_dicts]
        out[f"mean_{k}"] = float(np.mean(values))
        out[f"std_{k}"] = float(np.std(values))

    return out


# ============================================================
# OBJECTIVE FACTORIES
# ============================================================


def make_raw_objective(
    dataset: list[dict[str, Any]],
    base_config: FilamentConfig,
    search_space: dict[str, dict[str, Any]],
    objective_metric: str = "f0_5",
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        updates = {
            name: suggest_from_space(trial, name, spec)
            for name, spec in search_space.items()
        }
        cfg = replace(base_config, **updates)

        per_image_metrics: list[dict[str, float]] = []

        for item in dataset:
            pred_mask = build_raw_segmentation_mask(item["image"], cfg)
            metrics = compute_pixel_metrics(item["mask"], pred_mask)
            per_image_metrics.append(metrics)

        agg = aggregate_metric_dicts(per_image_metrics)

        for key, value in agg.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("config", cfg.__dict__)

        return agg[f"mean_{objective_metric}"]

    return objective


def make_cleanup_objective(
    dataset: list[dict[str, Any]],
    base_config: FilamentConfig,
    search_space: dict[str, dict[str, Any]],
    objective_metric: str = "f0_5",
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        updates = {
            name: suggest_from_space(trial, name, spec)
            for name, spec in search_space.items()
        }
        cfg = replace(base_config, **updates)

        per_image_metrics: list[dict[str, float]] = []

        for item in dataset:
            pred_mask = build_cleaned_mask(item["image"], cfg)
            metrics = compute_pixel_metrics(item["mask"], pred_mask)
            per_image_metrics.append(metrics)

        agg = aggregate_metric_dicts(per_image_metrics)

        for key, value in agg.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("config", cfg.__dict__)

        return agg[f"mean_{objective_metric}"]

    return objective


def make_presence_objective(
    dataset: list[dict[str, Any]],
    base_config: FilamentConfig,
    search_space: dict[str, dict[str, Any]],
    objective_metric: str = "f0_5",
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        updates = {
            name: suggest_from_space(trial, name, spec)
            for name, spec in search_space.items()
        }
        cfg = replace(base_config, **updates)

        y_true: list[bool] = []
        y_pred: list[bool] = []

        for item in dataset:
            pred_mask = build_cleaned_mask(item["image"], cfg)
            pred_present = derive_presence_from_mask(
                pred_mask,
                min_pixels_for_presence=cfg.min_pixels_for_presence,
            )

            y_true.append(item["filament_present"])
            y_pred.append(pred_present)

        metrics = compute_image_level_metrics(y_true, y_pred)

        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("config", cfg.__dict__)

        return metrics[objective_metric]

    return objective


# ============================================================
# STUDY RUNNERS
# ============================================================
#
# Progress / monitoring:
#   - ``show_progress_bar=True`` (default): Optuna tqdm bar for completed trials.
#   - Verbose logs: ``optuna.logging.set_verbosity(optuna.logging.INFO)`` before running.
#   - Persistent DB + UI: pass ``storage="sqlite:///path.db"`` then e.g. ``optuna-dashboard``.
#   - Custom per-trial prints: pass ``callbacks=[lambda study, trial: print(...)]``.


def run_raw_study(
    dataset: list[dict[str, Any]],
    base_config: FilamentConfig = BASE_CONFIG,
    search_space: dict[str, dict[str, Any]] = RAW_SEARCH_SPACE,
    n_trials: int = 100,
    study_name: str = "raw_segmentation",
    storage: str | None = None,
    seed: int = 42,
    *,
    show_progress_bar: bool = True,
    callbacks: list[Any] | None = None,
) -> optuna.Study:
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True if storage else False,
    )
    objective = make_raw_objective(
        dataset=dataset,
        base_config=base_config,
        search_space=search_space,
        objective_metric=RAW_OBJECTIVE_METRIC,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        callbacks=callbacks or [],
    )
    return study


def run_cleanup_study(
    dataset: list[dict[str, Any]],
    tuned_raw_config: FilamentConfig,
    search_space: dict[str, dict[str, Any]] = CLEANUP_SEARCH_SPACE,
    n_trials: int = 50,
    study_name: str = "cleanup",
    storage: str | None = None,
    seed: int = 42,
    *,
    show_progress_bar: bool = True,
    callbacks: list[Any] | None = None,
) -> optuna.Study:
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True if storage else False,
    )
    objective = make_cleanup_objective(
        dataset=dataset,
        base_config=tuned_raw_config,
        search_space=search_space,
        objective_metric=CLEANUP_OBJECTIVE_METRIC,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        callbacks=callbacks or [],
    )
    return study


def run_presence_study(
    dataset: list[dict[str, Any]],
    tuned_mask_config: FilamentConfig,
    search_space: dict[str, dict[str, Any]] = PRESENCE_SEARCH_SPACE,
    n_trials: int = 50,
    study_name: str = "presence",
    storage: str | None = None,
    seed: int = 42,
    *,
    show_progress_bar: bool = True,
    callbacks: list[Any] | None = None,
) -> optuna.Study:
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True if storage else False,
    )
    objective = make_presence_objective(
        dataset=dataset,
        base_config=tuned_mask_config,
        search_space=search_space,
        objective_metric=PRESENCE_OBJECTIVE_METRIC,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        callbacks=callbacks or [],
    )
    return study


# ============================================================
# CONFIG BUILDERS FROM BEST PARAMS
# ============================================================


def config_from_best_params(
    base_config: FilamentConfig, best_params: dict[str, Any]
) -> FilamentConfig:
    return replace(base_config, **best_params)


# ============================================================
# EXAMPLE MAIN
# ============================================================


def run():
    # ---------------------------
    # 1. Load labeled dataset
    # ---------------------------
    dataset = load_dataset_from_dirs(
        image_dir="data/separated_frames",
        mask_dir="data/annotated_data",
        mask_suffix="_mask.tif",
    )
    
    # Optional persistent SQLite storage for Optuna
    # storage = "sqlite:///filament_tuning.db"
    storage = None

    # ---------------------------
    # 2. Raw segmentation study
    # ---------------------------
    raw_study = run_raw_study(
        dataset=dataset,
        base_config=BASE_CONFIG,
        search_space=RAW_SEARCH_SPACE,
        n_trials=100,
        study_name="raw_segmentation",
        storage=storage,
        seed=42,
    )

    best_raw_config = config_from_best_params(BASE_CONFIG, raw_study.best_params)

    print("\n=== RAW STUDY ===")
    print("Best value:", raw_study.best_value)
    print("Best params:", raw_study.best_params)
    print("Best attrs:", raw_study.best_trial.user_attrs)

    # ---------------------------
    # 3. Cleanup study
    # ---------------------------
    cleanup_study = run_cleanup_study(
        dataset=dataset,
        tuned_raw_config=best_raw_config,
        search_space=CLEANUP_SEARCH_SPACE,
        n_trials=50,
        study_name="cleanup",
        storage=storage,
        seed=42,
    )

    best_mask_config = config_from_best_params(
        best_raw_config, cleanup_study.best_params
    )

    print("\n=== CLEANUP STUDY ===")
    print("Best value:", cleanup_study.best_value)
    print("Best params:", cleanup_study.best_params)
    print("Best attrs:", cleanup_study.best_trial.user_attrs)

    # ---------------------------
    # 4. Presence study
    # ---------------------------
    presence_study = run_presence_study(
        dataset=dataset,
        tuned_mask_config=best_mask_config,
        search_space=PRESENCE_SEARCH_SPACE,
        n_trials=50,
        study_name="presence",
        storage=storage,
        seed=42,
    )

    final_best_config = config_from_best_params(
        best_mask_config, presence_study.best_params
    )

    print("\n=== PRESENCE STUDY ===")
    print("Best value:", presence_study.best_value)
    print("Best params:", presence_study.best_params)
    print("Best attrs:", presence_study.best_trial.user_attrs)

    print("\n=== FINAL BEST CONFIG ===")
    print(final_best_config)
