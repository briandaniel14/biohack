import glob
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio.v2 import imread as imageio_imread
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops, regionprops_table, label
from skimage.morphology import skeletonize
from tifffile import imread, imwrite

from src.biohack.constants import FRAME_FILE_PATTERN


def remove_small_objects_from_labelmask(mask: np.ndarray, min_area: int) -> np.ndarray:
    props = pd.DataFrame(regionprops_table(mask, properties=["label", "area"]))
    if len(props) == 0:
        return np.zeros_like(mask, dtype=np.int32)

    keep_labels = props.loc[props["area"] >= min_area, "label"].astype(int).tolist()
    clean_mask = np.zeros_like(mask, dtype=np.int32)
    for new_label, old_label in enumerate(keep_labels, start=1):
        clean_mask[mask == old_label] = new_label
    return clean_mask


def segment_frame_cellpose(
    img: np.ndarray,
    model: Any,
    diameter: Optional[float] = None,
    channels: Optional[List[int]] = None,
) -> np.ndarray:
    eval_kwargs: Dict[str, Any] = {
        "diameter": diameter,
        "normalize": True,
        "do_3D": False,
    }
    if channels is not None:
        eval_kwargs["channels"] = channels

    result = model.eval(img, **eval_kwargs)
    masks = result[0] if isinstance(result, tuple) else result
    return masks


def build_pillar_reference_mask(
    gfp_path: str,
    gfp_model: Any,
    *,
    diameter_gfp: Optional[float] = None,
    channels_gfp: Optional[List[int]] = None,
    min_area_gfp: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    gfp_img = imread(gfp_path)
    gfp_masks = segment_frame_cellpose(
        gfp_img,
        model=gfp_model,
        diameter=diameter_gfp,
        channels=channels_gfp,
    )
    gfp_masks = remove_small_objects_from_labelmask(gfp_masks, min_area_gfp)
    return gfp_img, gfp_masks


def compute_iou(
    mask_a: np.ndarray, label_a: int, mask_b: np.ndarray, label_b: int
) -> float:
    a = mask_a == label_a
    b = mask_b == label_b
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def filter_bf_mask_with_gfp_reference(
    bf_mask: np.ndarray,
    gfp_reference_mask: np.ndarray,
    iou_threshold: float = 0.10,
) -> Tuple[np.ndarray, pd.DataFrame]:
    filtered_bf_masks = np.zeros_like(bf_mask, dtype=np.int32)
    kept_rows: List[Dict[str, Any]] = []
    new_label = 1

    gfp_labels = [r.label for r in regionprops(gfp_reference_mask)]

    for bf_region in regionprops(bf_mask):
        bf_label = bf_region.label
        best_iou = 0.0
        best_gfp_label: Optional[int] = None

        for gfp_label in gfp_labels:
            iou = compute_iou(bf_mask, bf_label, gfp_reference_mask, gfp_label)
            if iou > best_iou:
                best_iou = iou
                best_gfp_label = gfp_label

        keep = best_iou >= iou_threshold

        kept_rows.append(
            {
                "bf_label": int(bf_label),
                "best_gfp_label": None
                if best_gfp_label is None
                else int(best_gfp_label),
                "best_iou": float(best_iou),
                "keep": bool(keep),
            }
        )

        if keep:
            filtered_bf_masks[bf_mask == bf_label] = new_label
            new_label += 1

    debug_df = (
        pd.DataFrame(kept_rows).sort_values("bf_label")
        if len(kept_rows) > 0
        else pd.DataFrame()
    )
    return filtered_bf_masks, debug_df


def compute_overlap_fraction(
    prev_mask: np.ndarray,
    curr_mask: np.ndarray,
    prev_label: int,
    curr_label: int,
) -> float:
    prev_mask = np.asarray(prev_mask)
    curr_mask = np.asarray(curr_mask)

    prev_obj = prev_mask == prev_label
    curr_obj = curr_mask == curr_label

    prev_area = int(np.sum(prev_obj))
    curr_area = int(np.sum(curr_obj))

    if prev_area == 0 or curr_area == 0:
        return 0.0

    overlap = int(np.sum(prev_obj & curr_obj))
    return overlap / min(prev_area, curr_area)


def track_cells_one_movie(
    movie_name: str,
    frames_data: List[Dict[str, Any]],
    max_link_distance: int,
    min_overlap_fraction: float,
    bud_distance: int,
    bud_overlap_fraction: float,
) -> pd.DataFrame:
    next_cell_id = 1
    tracked_rows: List[pd.DataFrame] = []

    prev_df: Optional[pd.DataFrame] = None
    prev_mask: Optional[np.ndarray] = None

    for fd in frames_data:
        curr_df = fd["props"].copy()
        curr_mask = fd["mask"]

        if len(curr_df) == 0:
            prev_df = curr_df.copy()
            prev_mask = curr_mask.copy()
            continue

        curr_df["cell_ID"] = -1
        curr_df["mother_cell_ID"] = np.nan

        if prev_df is None or len(prev_df) == 0:
            for idx in curr_df.index:
                curr_df.loc[idx, "cell_ID"] = next_cell_id
                next_cell_id += 1
        else:
            prev_coords = prev_df[["centroid_y", "centroid_x"]].to_numpy()
            curr_coords = curr_df[["centroid_y", "centroid_x"]].to_numpy()

            dist_mat = cdist(prev_coords, curr_coords)
            cost_mat = dist_mat.copy()

            for i, prev_row in enumerate(prev_df.itertuples()):
                for j, curr_row in enumerate(curr_df.itertuples()):
                    overlap_frac = compute_overlap_fraction(
                        prev_mask,
                        curr_mask,
                        int(prev_row.label),
                        int(curr_row.label),
                    )

                    if overlap_frac >= min_overlap_fraction:
                        cost_mat[i, j] *= 0.25
                    elif dist_mat[i, j] > max_link_distance:
                        cost_mat[i, j] = 1e6

            row_ind, col_ind = linear_sum_assignment(cost_mat)

            assigned_curr = set()

            for i, j in zip(row_ind, col_ind):
                if cost_mat[i, j] >= 1e6:
                    continue

                prev_row = prev_df.iloc[i]
                curr_idx = curr_df.index[j]

                curr_df.loc[curr_idx, "cell_ID"] = int(prev_row["cell_ID"])
                assigned_curr.add(j)

            unassigned_curr = [j for j in range(len(curr_df)) if j not in assigned_curr]

            for j in unassigned_curr:
                curr_idx = curr_df.index[j]
                curr_label = curr_df.loc[curr_idx, "label"]
                curr_y = curr_df.loc[curr_idx, "centroid_y"]
                curr_x = curr_df.loc[curr_idx, "centroid_x"]

                curr_df.loc[curr_idx, "cell_ID"] = next_cell_id
                next_cell_id += 1

                best_mother: Optional[float] = None
                best_score = -1.0

                for i in range(len(prev_df)):
                    prev_row = prev_df.iloc[i]
                    prev_label = prev_row["label"]
                    prev_id = prev_row["cell_ID"]

                    d = float(
                        np.sqrt(
                            (curr_y - prev_row["centroid_y"]) ** 2
                            + (curr_x - prev_row["centroid_x"]) ** 2
                        )
                    )
                    overlap_frac = compute_overlap_fraction(
                        prev_mask,
                        curr_mask,
                        int(prev_label),
                        int(curr_label),
                    )

                    score = 0.0
                    if d <= bud_distance:
                        score += (bud_distance - d) / bud_distance
                    if overlap_frac >= bud_overlap_fraction:
                        score += overlap_frac * 2

                    if score > best_score and score > 0:
                        best_score = score
                        best_mother = float(prev_id)

                if best_mother is not None:
                    curr_df.loc[curr_idx, "mother_cell_ID"] = best_mother

        tracked_rows.append(curr_df)
        prev_df = curr_df.copy()
        prev_mask = curr_mask.copy()

    if len(tracked_rows) == 0:
        return pd.DataFrame()

    tracked_df = pd.concat(tracked_rows, ignore_index=True)
    tracked_df["movie"] = movie_name
    return tracked_df


def filter_pillar_tracks_v3_aggressive(
    cell_tracks_df: pd.DataFrame,
    *,
    min_track_len: int = 3,
    max_motion_std: float = 3.0,
    min_eccentricity: float = 0.68,
    min_aspect_ratio: float = 1.35,
    min_solidity: float = 0.94,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop cell tracks classified as imaging pillars (v3_aggressive rule).

    Aggregates geometry per ``(movie, cell_ID)``; a track is a pillar when it is
    long-lived, nearly motionless, elongated, and high-solidity. Returns the
    filtered per-frame table and a summary of removed tracks (for CSV export).
    """
    df = cell_tracks_df.copy()
    helper_cols = [
        "is_pillar",
        "n_frames",
        "track_motion_std",
        "median_eccentricity",
        "median_aspect_ratio",
        "median_solidity",
    ]
    df = df.drop(columns=[c for c in helper_cols if c in df.columns], errors="ignore")

    required_cols = [
        "movie",
        "cell_ID",
        "frame",
        "centroid_y",
        "centroid_x",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "solidity",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Pillar filter requires columns {required_cols}; missing: {missing}"
        )

    work = df.copy()
    numeric_cols = [
        "cell_ID",
        "frame",
        "centroid_y",
        "centroid_x",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "solidity",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if "area" in work.columns:
        work["area"] = pd.to_numeric(work["area"], errors="coerce")

    work = work.dropna(subset=["cell_ID"]).copy()
    work["cell_ID"] = work["cell_ID"].astype(int)

    work["aspect_ratio"] = (
        work["major_axis_length"] / work["minor_axis_length"].replace(0, np.nan)
    )

    agg_dict: Dict[str, Any] = {
        "frame": "nunique",
        "centroid_y": ["std", "median"],
        "centroid_x": ["std", "median"],
        "eccentricity": "median",
        "solidity": "median",
        "aspect_ratio": "median",
    }
    if "area" in work.columns:
        agg_dict["area"] = "median"

    track_summary = work.groupby(["movie", "cell_ID"]).agg(agg_dict)
    track_summary.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in track_summary.columns
    ]
    track_summary = track_summary.reset_index().rename(
        columns={
            "frame_nunique": "n_frames",
            "centroid_y_median": "median_centroid_y",
            "centroid_x_median": "median_centroid_x",
            "eccentricity_median": "median_eccentricity",
            "solidity_median": "median_solidity",
            "aspect_ratio_median": "median_aspect_ratio",
            "area_median": "median_area",
        }
    )

    track_summary["centroid_y_std"] = track_summary["centroid_y_std"].fillna(0)
    track_summary["centroid_x_std"] = track_summary["centroid_x_std"].fillna(0)
    track_summary["track_motion_std"] = np.sqrt(
        track_summary["centroid_y_std"] ** 2 + track_summary["centroid_x_std"] ** 2
    )

    track_summary["is_pillar"] = (
        (track_summary["n_frames"] >= min_track_len)
        & (track_summary["track_motion_std"] <= max_motion_std)
        & (track_summary["median_eccentricity"] >= min_eccentricity)
        & (track_summary["median_aspect_ratio"] >= min_aspect_ratio)
        & (track_summary["median_solidity"] >= min_solidity)
    )

    merge_cols = [
        "movie",
        "cell_ID",
        "is_pillar",
        "n_frames",
        "track_motion_std",
        "median_eccentricity",
        "median_aspect_ratio",
        "median_solidity",
    ]
    df_labeled = df.merge(
        track_summary[merge_cols],
        on=["movie", "cell_ID"],
        how="left",
    )
    df_labeled["is_pillar"] = df_labeled["is_pillar"].fillna(False)

    filtered_df = df_labeled.loc[~df_labeled["is_pillar"]].copy()
    drop_helpers = [
        "is_pillar",
        "n_frames",
        "track_motion_std",
        "median_eccentricity",
        "median_aspect_ratio",
        "median_solidity",
    ]
    filtered_df = filtered_df.drop(
        columns=[c for c in drop_helpers if c in filtered_df.columns],
        errors="ignore",
    )

    removed_summary = track_summary.loc[track_summary["is_pillar"]].copy()
    return filtered_df, removed_summary


def get_cell_mask_path(movie_name: str, frame: int, cell_mask_dir: str) -> str:
    return os.path.join(
        cell_mask_dir, f"{movie_name}_frame_{frame}_cellmask.tif"
    )


def load_existing_cell_mask(movie_name: str, frame: int, cell_mask_dir: str) -> np.ndarray:
    mask_path = get_cell_mask_path(movie_name, frame, cell_mask_dir)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing existing cell mask: {mask_path}")
    return imread(mask_path)


def clean_cell_mask(mask: np.ndarray, min_cell_area: int = 0) -> np.ndarray:
    mask = np.asarray(mask)

    if min_cell_area is None or min_cell_area <= 0:
        return mask.astype(np.int32)

    out = np.zeros_like(mask, dtype=np.int32)
    next_label = 1

    for label_id in np.unique(mask):
        if label_id == 0:
            continue
        obj = mask == label_id
        if obj.sum() >= min_cell_area:
            out[obj] = next_label
            next_label += 1

    return out


def get_cell_props(mask: np.ndarray, frame: int, movie_name: str) -> pd.DataFrame:
    mask = np.asarray(mask)

    props = regionprops_table(
        mask,
        properties=[
            "label",
            "area",
            "centroid",
            "bbox",
            "eccentricity",
            "solidity",
            "major_axis_length",
            "minor_axis_length",
        ],
    )

    df = pd.DataFrame(props)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "movie_name",
                "frame",
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
            ]
        )

    df = df.rename(
        columns={
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
            "bbox-0": "bbox_min_row",
            "bbox-1": "bbox_min_col",
            "bbox-2": "bbox_max_row",
            "bbox-3": "bbox_max_col",
        }
    )

    df.insert(0, "movie_name", movie_name)
    df.insert(1, "frame", frame)

    return df


def format_frame_token(frame: int) -> str:
    return f"{int(frame):04d}"


def collect_movie_frames(
    frame_dir: str, expected_channel: str
) -> Dict[str, List[Tuple[int, str]]]:
    
    movies_dict: Dict[str, List[Tuple[int, str]]] = {}

    for path in sorted(glob.glob(os.path.join(frame_dir, "*.tif*"))):
        fname = os.path.basename(path)
        match = FRAME_FILE_PATTERN.match(fname)

        if match is None:
            print(f"Skipping unmatched file: {fname}")
            continue

        channel = match.group("channel").upper()
        if channel != expected_channel.upper():
            continue

        movie_name = match.group("movie")
        frame = int(match.group("frame"))

        movies_dict.setdefault(movie_name, []).append((frame, path))

    for movie_name in movies_dict:
        movies_dict[movie_name] = sorted(
            movies_dict[movie_name], key=lambda x: x[0]
        )

    return movies_dict  


def skeleton_length_px(binary_mask: np.ndarray) -> float:
    skel = skeletonize(binary_mask > 0)
    return float(skel.sum())


def get_filament_props(
    filament_mask: np.ndarray, frame: int, movie_name: str
) -> pd.DataFrame:
    lab = label(filament_mask > 0)

    rows: List[pd.DataFrame] = []
    labels_present = [x for x in np.unique(lab) if x != 0]

    for i, region_label in enumerate(labels_present, start=1):
        reg = lab == region_label

        props = regionprops_table(
            reg.astype(np.uint8),
            properties=[
                "label",
                "area",
                "centroid",
                "bbox",
                "eccentricity",
                "solidity",
                "major_axis_length",
                "minor_axis_length",
            ],
        )

        df = pd.DataFrame(props)
        if len(df) == 0:
            continue

        df["movie"] = movie_name
        df["frame"] = frame
        df["mean_length_px"] = skeleton_length_px(reg)
        df["filament_local_label"] = i

        df = df.rename(
            columns={
                "centroid-0": "centroid_y",
                "centroid-1": "centroid_x",
                "bbox-0": "bbox_min_row",
                "bbox-1": "bbox_min_col",
                "bbox-2": "bbox_max_row",
                "bbox-3": "bbox_max_col",
            }
        )

        rows.append(df)

    if len(rows) == 0:
        return pd.DataFrame(
            columns=[
                "movie",
                "frame",
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
            ]
        )

    return pd.concat(rows, ignore_index=True)


def assign_filaments_to_cells(
    filament_mask: np.ndarray,
    cell_mask: np.ndarray,
    filament_props_df: pd.DataFrame,
    tracked_cells_frame_df: pd.DataFrame,
) -> pd.DataFrame:
    if len(filament_props_df) == 0:
        empty_df = filament_props_df.copy()
        empty_df["cell_ID"] = pd.Series(dtype="float")
        empty_df["overlap_pixels_with_cell"] = pd.Series(dtype="int")
        return empty_df

    out_rows: List[pd.Series] = []
    lab = label(filament_mask > 0)

    for _, row in filament_props_df.iterrows():
        filament_label = int(row["filament_local_label"])
        reg = lab == filament_label

        overlapped_cells = cell_mask[reg]
        overlapped_cells = overlapped_cells[overlapped_cells > 0]

        assigned_cell_id = np.nan
        overlap_pixels = 0

        if len(overlapped_cells) > 0:
            vals, counts = np.unique(overlapped_cells, return_counts=True)
            best_cell_label = vals[np.argmax(counts)]
            overlap_pixels = int(counts.max())

            matched = tracked_cells_frame_df.loc[
                tracked_cells_frame_df["label"] == best_cell_label
            ]

            if len(matched) > 0:
                assigned_cell_id = matched.iloc[0]["cell_ID"]

        row2 = row.copy()
        row2["cell_ID"] = assigned_cell_id
        row2["overlap_pixels_with_cell"] = overlap_pixels
        out_rows.append(row2)

    return pd.DataFrame(out_rows)


def get_filament_mask_path(
    movie_name: str, frame: int, filament_dir: str
) -> Optional[str]:
    mask_frame = int(frame) - 1
    base = f"{movie_name}_mask_{mask_frame}_1"

    for ext in ("", ".png", ".tif", ".tiff"):
        path = os.path.join(filament_dir, base + ext)
        if os.path.exists(path):
            return path

    return None


def build_filament_episodes(
    cell_tracks_with_filaments_df: pd.DataFrame, max_gap: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = cell_tracks_with_filaments_df.copy()

    if "filament_present" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    df = df[df["filament_present"] == 1].copy()

    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df["cell_ID"] = df["cell_ID"].astype(int)
    df = df.sort_values(["movie", "cell_ID", "frame"]).reset_index(drop=True)

    per_frame_rows: List[pd.Series] = []
    episode_rows: List[Dict[str, Any]] = []
    next_filament_id = 1

    for (movie, cell_id), grp in df.groupby(["movie", "cell_ID"]):
        grp = grp.sort_values("frame").reset_index(drop=True)

        current_episode_id: Optional[int] = None
        prev_frame: Optional[int] = None

        episode_frames: List[int] = []
        episode_lengths: List[float] = []
        episode_areas: List[float] = []

        for _, row in grp.iterrows():
            frame = int(row["frame"])

            new_episode = False
            if prev_frame is None:
                new_episode = True
            elif (frame - prev_frame) > (max_gap + 1):
                new_episode = True

            if new_episode:
                if current_episode_id is not None:
                    episode_rows.append(
                        {
                            "movie": movie,
                            "cell_ID": cell_id,
                            "filament_ID": current_episode_id,
                            "time_of_appearance": min(episode_frames),
                            "last_seen_frame": max(episode_frames),
                            "duration_frames": len(episode_frames),
                            "filament_count": 1,
                            "mean_length_px": float(np.mean(episode_lengths)),
                            "mean_area_px": float(np.mean(episode_areas)),
                        }
                    )

                current_episode_id = next_filament_id
                next_filament_id += 1
                episode_frames = []
                episode_lengths = []
                episode_areas = []

            assert current_episode_id is not None
            episode_frames.append(frame)
            episode_lengths.append(float(row["filament_mean_length_px"]))
            episode_areas.append(float(row["filament_area"]))

            row2 = row.copy()
            row2["filament_ID"] = current_episode_id
            row2["time_of_appearance"] = min(episode_frames)
            per_frame_rows.append(row2)

            prev_frame = frame

        if current_episode_id is not None and len(episode_frames) > 0:
            episode_rows.append(
                {
                    "movie": movie,
                    "cell_ID": cell_id,
                    "filament_ID": current_episode_id,
                    "time_of_appearance": min(episode_frames),
                    "last_seen_frame": max(episode_frames),
                    "duration_frames": len(episode_frames),
                    "filament_count": 1,
                    "mean_length_px": float(np.mean(episode_lengths)),
                    "mean_area_px": float(np.mean(episode_areas)),
                }
            )

    per_frame_with_ids_df = pd.DataFrame(per_frame_rows)
    episodes_df = pd.DataFrame(episode_rows)

    return per_frame_with_ids_df, episodes_df


def build_cell_lineage_table(cell_tracks_df: pd.DataFrame) -> pd.DataFrame:
    return (
        cell_tracks_df[["movie", "cell_ID", "mother_cell_ID"]]
        .drop_duplicates()
        .sort_values(["movie", "cell_ID"])
        .reset_index(drop=True)
    )


def build_cell_filament_summary(
    filament_episodes_df: pd.DataFrame, cell_lineage_df: pd.DataFrame
) -> pd.DataFrame:
    if len(filament_episodes_df) > 0:
        filament_count_per_cell = (
            filament_episodes_df.groupby(["movie", "cell_ID"])["filament_ID"]
            .nunique()
            .reset_index(name="filament_count")
        )

        first_filament_time_per_cell = (
            filament_episodes_df.groupby(["movie", "cell_ID"])["time_of_appearance"]
            .min()
            .reset_index(name="first_time_of_appearance")
        )

        mean_filament_length_per_cell = (
            filament_episodes_df.groupby(["movie", "cell_ID"])["mean_length_px"]
            .mean()
            .reset_index(name="mean_filament_length_px")
        )

        mean_filament_area_per_cell = (
            filament_episodes_df.groupby(["movie", "cell_ID"])["mean_area_px"]
            .mean()
            .reset_index(name="mean_filament_area_px")
        )

        total_filament_duration_per_cell = (
            filament_episodes_df.groupby(["movie", "cell_ID"])["duration_frames"]
            .sum()
            .reset_index(name="total_filament_duration_frames")
        )

        summary = filament_count_per_cell.merge(
            first_filament_time_per_cell,
            on=["movie", "cell_ID"],
            how="outer",
        )
        summary = summary.merge(
            mean_filament_length_per_cell, on=["movie", "cell_ID"], how="left"
        )
        summary = summary.merge(
            mean_filament_area_per_cell, on=["movie", "cell_ID"], how="left"
        )
        summary = summary.merge(
            total_filament_duration_per_cell, on=["movie", "cell_ID"], how="left"
        )
        summary = summary.merge(cell_lineage_df, on=["movie", "cell_ID"], how="left")
        return summary

    summary = cell_lineage_df.copy()
    summary["filament_count"] = 0
    summary["first_time_of_appearance"] = np.nan
    summary["mean_filament_length_px"] = np.nan
    summary["mean_filament_area_px"] = np.nan
    summary["total_filament_duration_frames"] = 0
    return summary


def merge_filament_ids_onto_cell_tracks(
    cell_tracks_with_filaments_df: pd.DataFrame,
    filaments_with_ids_df: pd.DataFrame,
) -> pd.DataFrame:
    if len(filaments_with_ids_df) > 0:
        filament_id_merge_df = filaments_with_ids_df[
            ["movie", "frame", "cell_ID", "filament_ID", "time_of_appearance"]
        ].drop_duplicates()

        merged = cell_tracks_with_filaments_df.merge(
            filament_id_merge_df,
            on=["movie", "frame", "cell_ID"],
            how="left",
        )
    else:
        merged = cell_tracks_with_filaments_df.copy()
        merged["filament_ID"] = np.nan
        merged["time_of_appearance"] = np.nan

    merged["frame_viewer"] = merged["frame"]
    return merged


def build_mask_stack_from_files(
    frame_list: List[Tuple[int, str]],
    get_mask_path_func: Callable[[int], Optional[str]],
    dtype: np.dtype = np.uint16,
) -> Tuple[Optional[np.ndarray], List[int]]:
    if len(frame_list) == 0:
        return None, []

    frames_sorted = [frame for frame, _ in frame_list]

    first_img = imread(frame_list[0][1])
    height, width = first_img.shape[:2]

    stack = np.zeros((len(frames_sorted), height, width), dtype=dtype)

    for i, (frame, _) in enumerate(frame_list):
        mask_path = get_mask_path_func(frame)

        if mask_path is None:
            continue

        mask = imread(mask_path)

        if mask.shape != (height, width):
            raise ValueError(
                f"Mask shape mismatch at frame {frame}: "
                f"expected {(height, width)}, got {mask.shape}"
            )

        stack[i] = mask.astype(dtype)

    return stack, frames_sorted


def print_pipeline_summary(
    cell_tracks_df: pd.DataFrame,
    cell_tracks_with_filaments_df: pd.DataFrame,
    filament_episodes_df: pd.DataFrame,
    cell_filament_summary_df: pd.DataFrame,
) -> None:
    print("Number of tracked cell-frame rows:", len(cell_tracks_df))
    print(
        "Number of cell-frame rows with merged filament info:",
        len(cell_tracks_with_filaments_df),
    )
    fp = cell_tracks_with_filaments_df
    if "filament_present" in fp.columns:
        print(
            "Number of filament-positive rows:",
            int(fp["filament_present"].sum()),
        )
    else:
        print("Number of filament-positive rows:", 0)
    print("Number of filament episodes:", len(filament_episodes_df))
    if "filament_count" in cell_filament_summary_df.columns:
        n_cells = int(cell_filament_summary_df["filament_count"].gt(0).sum())
    else:
        n_cells = 0
    print("Number of cells with at least one filament:", n_cells)


def plot_example_filament_presence(
    cell_tracks_with_filaments_and_ids_df: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (10, 4),
) -> None:
    example_df = cell_tracks_with_filaments_and_ids_df.copy()

    example_cells = (
        example_df.loc[example_df["filament_present"] == 1, "cell_ID"]
        .dropna()
        .unique()
    )

    if len(example_cells) == 0:
        print("No cells with filament detections found.")
        return

    example_cell_id = example_cells[0]
    plot_df = example_df[example_df["cell_ID"] == example_cell_id].sort_values(
        "frame"
    )

    plt.figure(figsize=figsize)
    plt.plot(plot_df["frame"], plot_df["filament_present"], marker="o")
    plt.xlabel("Frame")
    plt.ylabel("Filament present")
    plt.title(f"Cell {example_cell_id}: filament presence over time")
    plt.ylim(-0.1, 1.1)
    plt.show()


def plot_example_frame_with_masks(
    movie_frames_store: Dict[str, List[Dict[str, Any]]],
    movie_name: str,
    frame_list_index: int,
    filament_dir: str,
    *,
    figsize: Tuple[float, float] = (15, 5),
) -> None:
    fd = movie_frames_store[movie_name][frame_list_index]
    img = fd["image"]
    cell_mask = fd["mask"]
    frame = fd["frame"]

    filament_path = get_filament_mask_path(movie_name, frame, filament_dir)

    if filament_path is None:
        print("No filament mask for this frame.")
        return

    filament_mask = imageio_imread(filament_path)

    if filament_mask.ndim == 3:
        filament_mask = filament_mask[..., 0]

    filament_mask = (filament_mask > 0).astype(np.uint8)

    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Raw image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img, cmap="gray")
    plt.imshow(cell_mask, cmap="nipy_spectral", alpha=0.4)
    plt.title("Cells")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap="gray")
    plt.imshow(filament_mask, cmap="Reds", alpha=0.5)
    plt.title("Filament")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
