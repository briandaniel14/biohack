import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cellpose import models
from imageio.v2 import imread as imageio_imread
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops, regionprops_table, label
from skimage.morphology import skeletonize
from tifffile import imread, imwrite


# frame file pattern in our data:
# <movie_name>_frame_<frame_number>_<channel>.tif(f)
FRAME_FILE_PATTERN = re.compile(
    r"^(?P<movie>.+)_frame_(?P<frame>\d+)_(?P<channel>BF|GFP)\.tif{1,2}$",
    re.IGNORECASE,
)

@dataclass
class FilamentStatisticsConfig:
    input_dir: str = "data/separated_frames/"
    brightfield_dir: str = "data/separated_frames/brightfield/"
    gfp_dir: str = "data/separated_frames/gfp/"
    filament_dir: str = "data/separated_frames/filament/"
    cell_mask_dir: str = "data/separated_frames/cellpose_mask/"
    
    output_dir: str = "data/separated_frames/results/"

    # Set to True once to split raw 2-channel time stacks into per-frame TIFFs
    split_raw_stacks: bool = True

    # Channel order in raw stacks after reading:
    # channel 0 = brightfield, channel 1 = GFP
    brightfield_channel_index: int = 0
    gfp_channel_index: int = 1

    # Cellpose model type
    model_type_bf: str = "cyto3"
    model_type_gfp: str = "cyto3"
    diameter_bf: Optional[int] = None
    diameter_gfp: Optional[int] = None
    channels_bf: List[int] = field(default_factory=lambda: [0, 0])
    channels_gfp: List[int] = field(default_factory=lambda: [0, 0])
    use_gpu: bool = False

    # Keep BF objects only if they overlap the first GFP reference enough
    pillar_iou_threshold: float = 0.10

    # Optional tiny-object cleanup
    min_area_bf: int = 50
    min_area_gfp: int = 20

    # Reuse saved cell masks if they already exist
    use_existing_cell_masks: bool = False

    # Tracking settings
    max_link_distance: int = 60
    min_overlap_fraction: float = 0.15
    bud_distance: int = 35
    bud_overlap_fraction: float = 0.05

    max_filament_gap: int = 1


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


def split_stack(img: np.ndarray, *, verbose: bool = False) -> np.ndarray:
    if verbose:
        print("Original shape:", img.shape)

    # Case 1: already (T, C, Y, X)
    if img.ndim == 4 and img.shape[1] == 2:
        pass

    # Case 2: (T, Y, X, C)
    elif img.ndim == 4 and img.shape[-1] == 2:
        img = np.transpose(img, (0, 3, 1, 2))

    # Case 3: flattened (T*2, Y, X)
    elif img.ndim == 3:
        z, y, x = img.shape
        if z % 2 != 0:
            raise ValueError(f"Expected even number of slices, got {z}")
        t = z // 2
        img = img.reshape(t, 2, y, x)
    else:
        raise ValueError(f"Unsupported shape {img.shape}")

    if verbose:
        print("Converted to:", img.shape, "(T, C, Y, X)")
    return img


def split_raw_time_stacks(config: FilamentStatisticsConfig, *, verbose: bool = True) -> None:
    os.makedirs(config.brightfield_dir, exist_ok=True)
    os.makedirs(config.gfp_dir, exist_ok=True)

    files = [
        f
        for f in os.listdir(config.input_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ]
    if not files:
        raise RuntimeError(f"No TIFF files found in {config.input_dir}")

    bf_ch = config.brightfield_channel_index
    gfp_ch = config.gfp_channel_index

    for file in files:
        path = os.path.join(config.input_dir, file)
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
            bf_out = os.path.join(
                config.brightfield_dir, f"{stem}_frame_{t + 1:04d}_BF.tif"
            )
            gfp_out = os.path.join(
                config.gfp_dir, f"{stem}_frame_{t + 1:04d}_GFP.tif"
            )
            imwrite(bf_out, bf)
            imwrite(gfp_out, gfp)

        if verbose:
            print(f"Wrote {t_n} BF frames to {config.brightfield_dir}")
            print(f"Wrote {t_n} GFP frames to {config.gfp_dir}")

    if verbose:
        print("Done")


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
    diameter: Optional[int] = None,
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
    config: FilamentStatisticsConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    gfp_img = imread(gfp_path)
    gfp_masks = segment_frame_cellpose(
        gfp_img,
        model=gfp_model,
        diameter=config.diameter_gfp,
        channels=config.channels_gfp,
    )
    gfp_masks = remove_small_objects_from_labelmask(gfp_masks, config.min_area_gfp)
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
        frame = fd["frame"]
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

            assigned_prev = set()
            assigned_curr = set()

            for i, j in zip(row_ind, col_ind):
                if cost_mat[i, j] >= 1e6:
                    continue

                prev_row = prev_df.iloc[i]
                curr_idx = curr_df.index[j]

                curr_df.loc[curr_idx, "cell_ID"] = int(prev_row["cell_ID"])
                assigned_prev.add(i)
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


def get_cell_mask_path(
    movie_name: str, frame: int, cell_mask_dir: str
) -> str:
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


def run_filament_pipeline(
    config: FilamentStatisticsConfig,
    *,
    verbose: bool = True,
    plot_examples: bool = False,
) -> FilamentPipelineResult:
    """
    Run the full notebook workflow: optional stack split, Cellpose masks,
    tracking, filament assignment, episodes, summaries, and CSV exports.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cell_mask_dir, exist_ok=True)

    if config.split_raw_stacks:
        split_raw_time_stacks(config, verbose=verbose)

    movies = collect_movie_frames(config.brightfield_dir, expected_channel="BF")
    gfp_movies = collect_movie_frames(config.gfp_dir, expected_channel="GFP")

    if verbose:
        print(f"Found {len(movies)} BF movies and {len(gfp_movies)} GFP movies")
        for movie_name in sorted(movies):
            bf_count = len(movies[movie_name])
            gfp_count = len(gfp_movies.get(movie_name, []))
            print(f"  {movie_name}: BF={bf_count}, GFP={gfp_count}")

    if not movies:
        raise RuntimeError(
            f"No brightfield frame files matched the expected pattern in {config.brightfield_dir}. "
            "Expected names like <movie>_frame_0001_BF.tif"
        )

    model_bf: Optional[Any] = None
    model_gfp: Optional[Any] = None
    if not config.use_existing_cell_masks:
        model_bf = models.CellposeModel(
            gpu=config.use_gpu, model_type=config.model_type_bf
        )
        model_gfp = models.CellposeModel(
            gpu=config.use_gpu, model_type=config.model_type_gfp
        )

    all_cell_tracks: List[pd.DataFrame] = []
    movie_frames_store: Dict[str, List[Dict[str, Any]]] = {}

    for movie_name, frame_list in movies.items():
        if verbose:
            print(f"\nProcessing movie: {movie_name}")

        frames_data: List[Dict[str, Any]] = []

        for frame, path in frame_list:
            img = imread(path)

            out_mask_path = get_cell_mask_path(
                movie_name, frame, config.cell_mask_dir
            )

            if os.path.exists(out_mask_path):
                clean_mask = imread(out_mask_path)
                bf_masks = clean_mask.copy()
            else:
                if config.use_existing_cell_masks:
                    clean_mask = load_existing_cell_mask(
                        movie_name, frame, config.cell_mask_dir
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
                        diameter=config.diameter_bf,
                        channels=config.channels_bf,
                    )
                    bf_masks = clean_cell_mask(
                        bf_masks, min_cell_area=config.min_area_bf
                    )
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
            max_link_distance=config.max_link_distance,
            min_overlap_fraction=config.min_overlap_fraction,
            bud_distance=config.bud_distance,
            bud_overlap_fraction=config.bud_overlap_fraction,
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

    if "frame" in cell_tracks_df.columns:
        cell_tracks_df["frame_viewer"] = cell_tracks_df["frame"]
    elif verbose:
        print("Warning: cell_tracks_df is missing 'frame' column")

    tracks_path = os.path.join(config.output_dir, "cell_tracks_per_frame.csv")
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
                filament_dir=config.filament_dir,
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
            cell_tracks_with_filaments_df["filament_present"]
            .fillna(0)
            .astype(int)
        )

    if len(filaments_per_frame_df) > 0:
        filaments_per_frame_df = filaments_per_frame_df.copy()
        filaments_per_frame_df["frame_viewer"] = (
            filaments_per_frame_df["frame"] + 1
        )

    cell_tracks_with_filaments_df = cell_tracks_with_filaments_df.copy()
    cell_tracks_with_filaments_df["frame_viewer"] = (
        cell_tracks_with_filaments_df["frame"] + 1
    )

    filaments_csv = os.path.join(config.output_dir, "filaments_per_frame.csv")
    merged_csv = os.path.join(
        config.output_dir, "cell_tracks_per_frame_with_filaments.csv"
    )
    filaments_per_frame_df.to_csv(filaments_csv, index=False)
    cell_tracks_with_filaments_df.to_csv(merged_csv, index=False)
    if verbose:
        print("Saved:", filaments_csv)
        print("Saved:", merged_csv)

    filaments_with_ids_df, filament_episodes_df = build_filament_episodes(
        cell_tracks_with_filaments_df=cell_tracks_with_filaments_df,
        max_gap=config.max_filament_gap,
    )

    ids_csv = os.path.join(config.output_dir, "filaments_with_IDs_per_frame.csv")
    ep_csv = os.path.join(config.output_dir, "filament_episodes.csv")
    filaments_with_ids_df.to_csv(ids_csv, index=False)
    filament_episodes_df.to_csv(ep_csv, index=False)
    if verbose:
        print("Saved:", ids_csv)
        print("Saved:", ep_csv)

    cell_lineage_df = build_cell_lineage_table(cell_tracks_df)

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

    lin_csv = os.path.join(
        config.output_dir, "filament_episodes_with_lineage.csv"
    )
    filament_episodes_with_lineage_df.to_csv(lin_csv, index=False)
    if verbose:
        print("Saved:", lin_csv)

    cell_filament_summary_df = build_cell_filament_summary(
        filament_episodes_df, cell_lineage_df
    )

    summary_csv = os.path.join(config.output_dir, "cell_filament_summary.csv")
    cell_filament_summary_df.to_csv(summary_csv, index=False)
    if verbose:
        print("Saved:", summary_csv)

    cell_tracks_with_filaments_and_ids_df = merge_filament_ids_onto_cell_tracks(
        cell_tracks_with_filaments_df,
        filaments_with_ids_df,
    )

    both_csv = os.path.join(
        config.output_dir, "cell_tracks_per_frame_with_filaments_and_ids.csv"
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
                config.filament_dir,
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
    )
