from dataclasses import dataclass, field, fields
from typing import Any, List, Mapping, Optional, cast


def _coerce_optional_float(value: Any) -> Optional[float]:
    """
    YAML often maps ``diameter_bf: None`` to the string ``\"None\"`` (not null).
    Cellpose expects a real number or None to auto-estimate.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in ("none", "null", "~", "nan"):
            return None
        try:
            return float(s)
        except ValueError as exc:
            raise ValueError(
                f"Expected a numeric diameter or null/empty, got {value!r}"
            ) from exc
    if isinstance(value, bool):
        raise TypeError("diameter must be numeric or null, not a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Unsupported diameter type: {type(value).__name__}")


@dataclass()
class ExperimentConfig:
    """
    Configuration for the experiment.

    Holds information (1) about the experiment itself,
    (2) parameters for the filament segmentation, and (3)
    parameters for the cell tracking.
    """

    run_name: str
    
    # directory containing the dataset, i.e. frames for brightfield and gfp
    # subdirectoies: uid/brightfield, uid/gfp
    dataset_directory: str
    
    # used to point the segmentation engine to the concrete data directory
    # i.e data/<uid_dataset>/...
    results_directory: str

    verbose: bool = True
    max_workers: int = 6

    # ----------------------------------------------------------------------
    # --------------- Filament segmentation parameters ---------------------
    # ----------------------------------------------------------------------
    clip_low_percentile: float = 0.0
    clip_high_percentile: float = 100.0

    gaussian_sigma: float = 1.0

    foreground_percentile: float = 92.0

    local_block_size: int = 35
    local_offset: float = -0.01

    frangi_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)
    frangi_threshold_percentile: float = 85.0

    min_object_size: int = 20
    min_pixels_for_presence: int = 20

    figure_dpi: int = 140
    cmap: str = "gray"

    # ----------------------------------------------------------------------
    # --------------- Cell tracking parameters -----------------------------
    # ----------------------------------------------------------------------

    model_type_bf: str = "cyto3"
    model_type_gfp: str = "cyto3"
    diameter_bf: Optional[float] = None
    diameter_gfp: Optional[float] = None
    channels_bf: List[int] = field(default_factory=lambda: [0, 0])
    channels_gfp: List[int] = field(default_factory=lambda: [0, 0])
    use_gpu: bool = False
    pillar_iou_threshold: float = 0.10
    min_area_bf: int = 50
    min_area_gfp: int = 20
    use_existing_cell_masks: bool = False
    max_link_distance: int = 60
    min_overlap_fraction: float = 0.15
    bud_distance: int = 35
    bud_overlap_fraction: float = 0.05
    max_filament_gap: int = 1

    # Post-tracking: remove static elongated tracks (imaging pillars) before filaments.
    remove_pillar_tracks: bool = False
    pillar_v3_min_track_frames: int = 3
    pillar_v3_max_motion_std: float = 3.0
    pillar_v3_min_eccentricity: float = 0.68
    pillar_v3_min_aspect_ratio: float = 1.35
    pillar_v3_min_solidity: float = 0.94

    def __post_init__(self) -> None:
        self.diameter_bf = _coerce_optional_float(self.diameter_bf)
        self.diameter_gfp = _coerce_optional_float(self.diameter_gfp)
        if isinstance(self.frangi_sigmas, list):
            self.frangi_sigmas = tuple(
                float(x) for x in cast(list[Any], self.frangi_sigmas)
            )

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "ExperimentConfig":
        """Alias for :func:`experiment_config_from_mapping`."""
        return experiment_config_from_mapping(mapping)


def experiment_config_from_mapping(mapping: Mapping[str, Any]) -> ExperimentConfig:
    """Build :class:`ExperimentConfig` from a YAML/dict, ignoring unknown keys."""
    known = {f.name for f in fields(ExperimentConfig)}
    kwargs = {k: v for k, v in mapping.items() if k in known}
    return ExperimentConfig(**kwargs)
