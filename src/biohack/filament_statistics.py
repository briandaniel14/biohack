"""Cell segmentation, tracking, filament measurement, and episode summaries."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from cellpose import models
from imageio.v2 import imread as imageio_imread
from tifffile import imread, imwrite

from src.biohack.experiment_config import ExperimentConfig
from src.biohack.statistics_helper import (
    assign_filaments_to_cells,
    build_cell_filament_summary,
    build_cell_lineage_table,
    build_filament_episodes,
    clean_cell_mask,
    collect_movie_frames,
    filter_pillar_tracks_v3_aggressive,
    get_cell_mask_path,
    get_cell_props,
    get_filament_mask_path,
    get_filament_props,
    load_existing_cell_mask,
    merge_filament_ids_onto_cell_tracks,
    plot_example_filament_presence,
    plot_example_frame_with_masks,
    print_pipeline_summary,
    segment_frame_cellpose,
    track_cells_one_movie,
)
from src.biohack.constants import (
    RUN_SUBDIR_BRIGHTFIELD,
    RUN_SUBDIR_GFP,
    RUN_SUBDIR_FILAMENT_MASK,
    RUN_SUBDIR_CELLPOSE_MASK,
    RUN_SUBDIR_STATISTICS,
    REMOVED_PILLAR_TRACKS_CSV_NAME,
)

def resolve_run_paths(
    experiment: ExperimentConfig, run_id: str
) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Resolve per-run directories for :func:`run_filament_pipeline`.

    Expected layout (from filament detection + this pipeline):

    - ``brightfield/``, ``gfp/``, ``filament_mask/`` from detection
    - ``cellpose_mask/`` written here when segmenting cells
    - ``statistics/`` CSV outputs from this pipeline
    """
    results_root = Path(experiment.results_directory)
    run_root = results_root / run_id

    brightfield_dir = run_root / RUN_SUBDIR_BRIGHTFIELD
    gfp_dir = run_root / RUN_SUBDIR_GFP
    filament_dir = run_root / RUN_SUBDIR_FILAMENT_MASK
    cell_mask_dir = run_root / RUN_SUBDIR_CELLPOSE_MASK
    output_dir = run_root / RUN_SUBDIR_STATISTICS
    return brightfield_dir, gfp_dir, filament_dir, cell_mask_dir, output_dir


@dataclass
class FilamentPipelineResult:
    cell_tracks_df: pd.DataFrame
    movie_frames_store: Dict[str, List[Dict[str, Any]]]
    filaments_per_frame_df: pd.DataFrame
    cell_tracks_with_filaments_df: pd.DataFrame
    filaments_with_ids_df: pd.DataFrame
    filament_episodes_df: pd.DataFrame
    filament_episodes_with_lineage_df: pd.DataFrame
    cell_filament_summary_df: pd.DataFrame
    cell_tracks_with_filaments_and_ids_df: pd.DataFrame
    run_root: Path


def run_filament_pipeline(
    experiment: ExperimentConfig,
    run_id: str,
    *,
    verbose: Optional[bool] = None,
    plot_examples: bool = False,
) -> FilamentPipelineResult:
    """
    Full workflow on one detection run: Cellpose cell masks, tracking, filament
    tables, episodes, CSVs under ``<results_directory>/<run_id>/statistics/``.

    Input frames and filament masks are read from the run snapshot produced by
    :func:`biohack.image_detection.process_directory` (``brightfield/``, ``gfp/``,
    ``filament_mask/``). Only ``experiment.results_directory`` and segmentation /
    tracking fields are used; ``dataset_directory`` is ignored here.

    If ``experiment.remove_pillar_tracks`` is true, static elongated tracks are
    removed after tracking and before filament assignment; summaries are written
    to ``statistics/removed_pillar_tracks_v3.csv``.
    """
    exp = experiment
    verbose = exp.verbose

    bf_p, gfp_p, fil_p, cell_p, out_p = resolve_run_paths(exp, run_id)

    print(bf_p, gfp_p, fil_p, cell_p, out_p)

    brightfield_dir = str(bf_p)
    gfp_dir = str(gfp_p)
    filament_dir = str(fil_p)
    cell_mask_dir = str(cell_p)
    output_dir = str(out_p)
    run_root = bf_p.parent

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cell_mask_dir, exist_ok=True)

    movies = collect_movie_frames(brightfield_dir, expected_channel="BF")
    gfp_movies = collect_movie_frames(gfp_dir, expected_channel="GFP")

    if verbose:
        print(f"Run: {run_id}  root: {run_root}")
        print(f"Found {len(movies)} BF movies and {len(gfp_movies)} GFP movies")
        for movie_name in sorted(movies):
            bf_count = len(movies[movie_name])
            gfp_count = len(gfp_movies.get(movie_name, []))
            print(f"  {movie_name}: BF={bf_count}, GFP={gfp_count}")

    if not movies:
        raise RuntimeError(
            f"No brightfield frame files matched the expected pattern in {brightfield_dir}. "
            "Expected names like <movie>_frame_0001_BF.tif"
        )

    model_bf: Optional[Any] = None
    if not exp.use_existing_cell_masks:
        model_bf = models.CellposeModel(gpu=exp.use_gpu, model_type=exp.model_type_bf)

    all_cell_tracks: List[pd.DataFrame] = []
    movie_frames_store: Dict[str, List[Dict[str, Any]]] = {}

    for movie_name, frame_list in movies.items():
        if verbose:
            print(f"\nProcessing movie: {movie_name}")

        frames_data: List[Dict[str, Any]] = []

        for frame, path in frame_list:
            img = imread(path)
            print("processing frame", frame)

            out_mask_path = get_cell_mask_path(movie_name, frame, cell_mask_dir)

            if os.path.exists(out_mask_path):
                clean_mask = imread(out_mask_path)
                bf_masks = clean_mask.copy()
            else:
                if exp.use_existing_cell_masks:
                    clean_mask = load_existing_cell_mask(
                        movie_name, frame, cell_mask_dir
                    )
                    bf_masks = clean_mask.copy()
                else:
                    if model_bf is None:
                        raise RuntimeError(
                            "Cellpose model is required when not using existing masks"
                        )
                    bf_masks = segment_frame_cellpose(
                        img,
                        model=model_bf,
                        diameter=exp.diameter_bf,
                        channels=exp.channels_bf,
                    )
                    bf_masks = clean_cell_mask(bf_masks, min_cell_area=exp.min_area_bf)
                    clean_mask = bf_masks.copy()

                imwrite(out_mask_path, clean_mask.astype(np.uint16))

            cell_props = get_cell_props(clean_mask, frame, movie_name)

            frames_data.append(
                {
                    "frame": frame,
                    "image": img,
                    "mask": clean_mask,
                    "props": cell_props,
                }
            )

        movie_frames_store[movie_name] = frames_data

        tracked_df = track_cells_one_movie(
            movie_name,
            frames_data,
            max_link_distance=exp.max_link_distance,
            min_overlap_fraction=exp.min_overlap_fraction,
            bud_distance=exp.bud_distance,
            bud_overlap_fraction=exp.bud_overlap_fraction,
        )

        if tracked_df is not None and not tracked_df.empty:
            all_cell_tracks.append(tracked_df)
        elif verbose:
            print(f"Warning: no tracked cells produced for {movie_name}")

    cell_tracks_df = (
        pd.concat(all_cell_tracks, ignore_index=True)
        if len(all_cell_tracks) > 0
        else pd.DataFrame()
    )

    if cell_tracks_df.empty:
        raise RuntimeError(
            "No cell tracks were produced for any movie; check inputs and segmentation."
        )

    if exp.remove_pillar_tracks:
        rows_before = len(cell_tracks_df)
        cell_tracks_df, removed_pillar_summary = filter_pillar_tracks_v3_aggressive(
            cell_tracks_df,
            min_track_len=exp.pillar_v3_min_track_frames,
            max_motion_std=exp.pillar_v3_max_motion_std,
            min_eccentricity=exp.pillar_v3_min_eccentricity,
            min_aspect_ratio=exp.pillar_v3_min_aspect_ratio,
            min_solidity=exp.pillar_v3_min_solidity,
        )
        removed_pillar_path = os.path.join(output_dir, REMOVED_PILLAR_TRACKS_CSV_NAME)
        removed_pillar_summary.to_csv(removed_pillar_path, index=False)
        rows_after = len(cell_tracks_df)
        if verbose:
            print(f"Saved: {removed_pillar_path}")
            print(
                f"Pillar removal (v3): {len(removed_pillar_summary)} tracks removed, "
                f"{rows_before - rows_after} rows dropped, {rows_after} rows remaining"
            )
        if cell_tracks_df.empty:
            raise RuntimeError(
                "All cell tracks were classified as pillars and removed; "
                "loosen pillar_v3_* thresholds or disable remove_pillar_tracks."
            )

    if "frame" in cell_tracks_df.columns:
        cell_tracks_df["frame_viewer"] = cell_tracks_df["frame"]
    elif verbose:
        print("Warning: cell_tracks_df is missing 'frame' column")

    tracks_path = os.path.join(output_dir, "cell_tracks_per_frame.csv")
    cell_tracks_df.to_csv(tracks_path, index=False)
    if verbose:
        print("\nSaved:", tracks_path)

    all_filaments: List[pd.DataFrame] = []

    for movie_name, frames_data in movie_frames_store.items():
        if verbose:
            print(f"Loading filaments for {movie_name}")

        for fd in frames_data:
            frame = fd["frame"]
            cell_mask = fd["mask"]

            filament_path = get_filament_mask_path(
                movie_name=movie_name,
                frame=frame,
                filament_dir=filament_dir,
            )

            if filament_path is None:
                continue

            filament_mask = imageio_imread(filament_path)

            if filament_mask.ndim == 3:
                filament_mask = filament_mask[..., 0]

            filament_mask = (filament_mask > 0).astype(np.uint8)

            filament_props = get_filament_props(
                filament_mask=filament_mask,
                frame=frame,
                movie_name=movie_name,
            )

            tracked_cells_frame_df = cell_tracks_df[
                (cell_tracks_df["movie"] == movie_name)
                & (cell_tracks_df["frame"] == frame)
            ].copy()

            assigned_df = assign_filaments_to_cells(
                filament_mask=filament_mask,
                cell_mask=cell_mask,
                filament_props_df=filament_props,
                tracked_cells_frame_df=tracked_cells_frame_df,
            )

            if len(assigned_df) > 0:
                all_filaments.append(assigned_df)

    if len(all_filaments) > 0:
        filaments_per_frame_df = pd.concat(all_filaments, ignore_index=True)
    else:
        filaments_per_frame_df = pd.DataFrame(
            columns=[
                "movie",
                "frame",
                "cell_ID",
                "label",
                "area",
                "centroid_y",
                "centroid_x",
                "bbox_min_row",
                "bbox_min_col",
                "bbox_max_row",
                "bbox_max_col",
                "eccentricity",
                "solidity",
                "major_axis_length",
                "minor_axis_length",
                "mean_length_px",
                "filament_local_label",
                "overlap_pixels_with_cell",
            ]
        )

    if len(filaments_per_frame_df) > 0:
        filaments_per_frame_df = (
            filaments_per_frame_df.sort_values(
                ["movie", "frame", "cell_ID", "area"],
                ascending=[True, True, True, False],
            )
            .drop_duplicates(subset=["movie", "frame", "cell_ID"], keep="first")
            .copy()
        )

    filament_merge_df = (
        filaments_per_frame_df.rename(
            columns={
                "label": "filament_label",
                "area": "filament_area",
                "centroid_y": "filament_centroid_y",
                "centroid_x": "filament_centroid_x",
                "bbox_min_row": "filament_bbox_min_row",
                "bbox_min_col": "filament_bbox_min_col",
                "bbox_max_row": "filament_bbox_max_row",
                "bbox_max_col": "filament_bbox_max_col",
                "eccentricity": "filament_eccentricity",
                "solidity": "filament_solidity",
                "major_axis_length": "filament_major_axis_length",
                "minor_axis_length": "filament_minor_axis_length",
                "mean_length_px": "filament_mean_length_px",
            }
        )
        if len(filaments_per_frame_df) > 0
        else pd.DataFrame()
    )

    if len(filament_merge_df) > 0:
        filament_merge_df["filament_present"] = 1

    merge_cols = [
        "movie",
        "frame",
        "cell_ID",
        "filament_present",
        "filament_local_label",
        "filament_label",
        "filament_area",
        "filament_centroid_y",
        "filament_centroid_x",
        "filament_bbox_min_row",
        "filament_bbox_min_col",
        "filament_bbox_max_row",
        "filament_bbox_max_col",
        "filament_eccentricity",
        "filament_solidity",
        "filament_major_axis_length",
        "filament_minor_axis_length",
        "filament_mean_length_px",
        "overlap_pixels_with_cell",
    ]

    if len(filament_merge_df) > 0:
        filament_merge_df = filament_merge_df[merge_cols]
    else:
        filament_merge_df = pd.DataFrame(columns=merge_cols)

    cell_tracks_with_filaments_df = cell_tracks_df.merge(
        filament_merge_df,
        on=["movie", "frame", "cell_ID"],
        how="left",
    )

    if "filament_present" in cell_tracks_with_filaments_df.columns:
        cell_tracks_with_filaments_df["filament_present"] = (
            cell_tracks_with_filaments_df["filament_present"].fillna(0).astype(int)
        )

    if len(filaments_per_frame_df) > 0:
        filaments_per_frame_df = filaments_per_frame_df.copy()
        filaments_per_frame_df["frame_viewer"] = filaments_per_frame_df["frame"] + 1

    cell_tracks_with_filaments_df = cell_tracks_with_filaments_df.copy()
    cell_tracks_with_filaments_df["frame_viewer"] = (
        cell_tracks_with_filaments_df["frame"] + 1
    )

    filaments_csv = os.path.join(output_dir, "filaments_per_frame.csv")
    merged_csv = os.path.join(output_dir, "cell_tracks_per_frame_with_filaments.csv")
    filaments_per_frame_df.to_csv(filaments_csv, index=False)
    cell_tracks_with_filaments_df.to_csv(merged_csv, index=False)
    if verbose:
        print("Saved:", filaments_csv)
        print("Saved:", merged_csv)

    filaments_with_ids_df, filament_episodes_df = build_filament_episodes(
        cell_tracks_with_filaments_df=cell_tracks_with_filaments_df,
        max_gap=exp.max_filament_gap,
    )

    ids_csv = os.path.join(output_dir, "filaments_with_IDs_per_frame.csv")
    ep_csv = os.path.join(output_dir, "filament_episodes.csv")
    filaments_with_ids_df.to_csv(ids_csv, index=False)
    filament_episodes_df.to_csv(ep_csv, index=False)
    if verbose:
        print("Saved:", ids_csv)
        print("Saved:", ep_csv)

    cell_lineage_df = build_cell_lineage_table(cell_tracks_df)

    if len(cell_lineage_df) > 0 and len(filament_episodes_df) > 0:
        filament_episodes_with_lineage_df = filament_episodes_df.merge(
            cell_lineage_df,
            on=["movie", "cell_ID"],
            how="left",
        )
        filament_episodes_with_lineage_df["time_of_appearance_viewer"] = (
            filament_episodes_with_lineage_df["time_of_appearance"]
        )
        filament_episodes_with_lineage_df["last_seen_frame_viewer"] = (
            filament_episodes_with_lineage_df["last_seen_frame"]
        )
    else:
        filament_episodes_with_lineage_df = pd.DataFrame()	


    lin_csv = os.path.join(output_dir, "filament_episodes_with_lineage.csv")
    filament_episodes_with_lineage_df.to_csv(lin_csv, index=False)
    if verbose:
        print("Saved:", lin_csv)

    cell_filament_summary_df = build_cell_filament_summary(
        filament_episodes_df, cell_lineage_df
    )

    summary_csv = os.path.join(output_dir, "cell_filament_summary.csv")
    cell_filament_summary_df.to_csv(summary_csv, index=False)
    if verbose:
        print("Saved:", summary_csv)

    cell_tracks_with_filaments_and_ids_df = merge_filament_ids_onto_cell_tracks(
        cell_tracks_with_filaments_df,
        filaments_with_ids_df,
    )

    both_csv = os.path.join(
        output_dir, "cell_tracks_per_frame_with_filaments_and_ids.csv"
    )
    cell_tracks_with_filaments_and_ids_df.to_csv(both_csv, index=False)
    if verbose:
        print("Saved:", both_csv)

    print_pipeline_summary(
        cell_tracks_df,
        cell_tracks_with_filaments_df,
        filament_episodes_df,
        cell_filament_summary_df,
    )

    if plot_examples:
        plot_example_filament_presence(cell_tracks_with_filaments_and_ids_df)
        if movie_frames_store:
            first_movie = next(iter(movie_frames_store))
            n_frames = len(movie_frames_store[first_movie])
            idx = min(22, n_frames - 1) if n_frames else 0
            plot_example_frame_with_masks(
                movie_frames_store,
                first_movie,
                idx,
                filament_dir,
            )

    return FilamentPipelineResult(
        cell_tracks_df=cell_tracks_df,
        movie_frames_store=movie_frames_store,
        filaments_per_frame_df=filaments_per_frame_df,
        cell_tracks_with_filaments_df=cell_tracks_with_filaments_df,
        filaments_with_ids_df=filaments_with_ids_df,
        filament_episodes_df=filament_episodes_df,
        filament_episodes_with_lineage_df=filament_episodes_with_lineage_df,
        cell_filament_summary_df=cell_filament_summary_df,
        cell_tracks_with_filaments_and_ids_df=cell_tracks_with_filaments_and_ids_df,
        run_root=run_root,
    )
