"""
Flask API server for the filament-detection frontend.

Endpoints
---------
POST /api/upload        Upload a multi-frame TIFF → split → create dataset
POST /api/run           Run the detection pipeline on a dataset (with params)
GET  /api/status/<id>   Poll a running job
GET  /api/datasets      List available datasets (reads datasets.json)
GET  /data/<path>       Serve static dataset files (images, masks, etc.)
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, after_this_request, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # biohack/
DATA_DIR = BASE_DIR / "server_data"                        # served as /data/
UPLOAD_TMP = BASE_DIR / "server_uploads"
ORIG_TIFFS = BASE_DIR / "server_originals"                 # original uploaded TIFFs
SAMPLE_CSV = BASE_DIR / "test" / "cell_tracks_per_frame_with_filaments_and_ids.csv"
ALLOWED_EXTENSIONS = {".tif", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024      # 500 MB

# In-memory job tracker  {job_id: {status, step, error, dataset_id}}
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_INTERVAL_MIN = 15

def _format_timespan(n_frames: int) -> str:
    total_min = (n_frames - 1) * _FRAME_INTERVAL_MIN
    h, m = divmod(total_min, 60)
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"


def _datasets_json_path() -> Path:
    return DATA_DIR / "datasets.json"


def _read_datasets() -> list[dict]:
    p = _datasets_json_path()
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def _write_datasets(ds: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_datasets_json_path(), "w") as f:
        json.dump(ds, f, indent=2)


def _set_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(fields)


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def _write_dataset_zip(
    zip_path: Path, ds_dir: Path, mode: str, screenshots: list | None = None
) -> None:
    import base64
    import zipfile

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_f in sorted(ds_dir.glob("*.csv")):
            zf.write(csv_f, csv_f.name)

        summary_p = ds_dir / "summary.json"
        if summary_p.exists():
            zf.write(summary_p, "summary.json")

        if mode == "all":
            for subdir in ("raw", "masks", "diagnostics"):
                sub = ds_dir / subdir
                if not sub.is_dir():
                    continue
                for fp in sorted(sub.iterdir()):
                    if fp.is_file():
                        zf.write(fp, f"{subdir}/{fp.name}")

        # Include client-side screenshots (charts, table)
        if screenshots:
            for shot in screenshots:
                name = shot.get("name", "")
                data = shot.get("data", "")
                if not name or not data:
                    continue
                # Sanitise name: only allow simple filenames in known subdirs
                safe = Path(name).name if "/" not in name else f"{Path(name).parent.name}/{Path(name).name}"
                if safe and not safe.startswith(("..", "/")):
                    zf.writestr(safe, base64.b64decode(data))


def _frame_to_uint8_like_pipeline_raw_panel(frame: np.ndarray) -> np.ndarray:
    """Match ``plot_pipeline_results`` \"Raw image\" panel (``ensure_grayscale_float`` + default ``imshow`` norm).

    Matplotlib scales float grayscale with vmin/vmax = data min/max; this is not the same as
    a fixed percentile stretch (e.g. 0.5–99.5).
    """
    from skimage import util as skutil

    if frame.ndim == 3:
        img = skutil.img_as_float(frame)
        img = img[..., :3].mean(axis=-1)
    else:
        img = skutil.img_as_float(frame)
    lo = float(np.min(img))
    hi = float(np.max(img))
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    stretched = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return skutil.img_as_ubyte(stretched)


def _split_tiff_to_pngs(tiff_path: Path, out_dir: Path, dataset_id: str) -> int:
    """Split a multi-frame TIFF into individual PNG frames for browser display.

    Handles both single-channel and 2-channel (BF+GFP) stacks.
    For 2-channel data, displays channel 1 (GFP).
    Also saves the original TIFF for later pipeline use.
    Returns frame count.
    """
    import shutil
    import tifffile
    from skimage import io as skio

    stack = tifffile.imread(str(tiff_path))

    # Detect and handle multi-channel stacks → extract GFP (channel 1)
    if stack.ndim == 4 and stack.shape[1] == 2:
        # (T, C, Y, X) — take GFP channel
        display_stack = stack[:, 1, :, :]
    elif stack.ndim == 4 and stack.shape[-1] == 2:
        # (T, Y, X, C)
        display_stack = stack[:, :, :, 1]
    elif stack.ndim == 3:
        z = stack.shape[0]
        if z % 2 == 0 and z >= 4:
            # Likely interleaved 2-channel: reshape and take GFP
            t = z // 2
            reshaped = stack.reshape(t, 2, stack.shape[1], stack.shape[2])
            display_stack = reshaped[:, 1, :, :]
        else:
            display_stack = stack
    elif stack.ndim == 2:
        display_stack = stack[np.newaxis, ...]
    else:
        display_stack = stack

    if display_stack.ndim == 2:
        display_stack = display_stack[np.newaxis, ...]

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(display_stack):
        frame_8 = _frame_to_uint8_like_pipeline_raw_panel(frame)
        skio.imsave(str(out_dir / f"frame_{i:03d}.png"), frame_8, check_contrast=False)

    # Preserve original TIFF for pipeline use
    orig_dir = ORIG_TIFFS / dataset_id
    orig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(tiff_path), str(orig_dir / tiff_path.name))

    return len(display_stack)


def _run_pipeline_job(
    job_id: str,
    dataset_id: str,
    raw_dir: Path,
    params: dict[str, Any],
) -> None:
    """Background worker: runs the full pipeline (detection + statistics), updates job status."""
    import sys
    sys.path.insert(0, str(BASE_DIR))

    try:
        _set_job(job_id, status="running", dataset_id=dataset_id, step="Initializing pipeline...")

        ds_dir = DATA_DIR / dataset_id
        pipeline_output = ds_dir / "pipeline_output"
        pipeline_output.mkdir(parents=True, exist_ok=True)

        # --- Locate original TIFF ---
        orig_dir = ORIG_TIFFS / dataset_id
        orig_tiff = None
        if orig_dir.is_dir():
            tiffs = sorted(orig_dir.glob("*.tif")) + sorted(orig_dir.glob("*.tiff"))
            if tiffs:
                orig_tiff = tiffs[0]

        if orig_tiff is None:
            # Fallback: run old-style pipeline on PNGs
            _run_legacy_pipeline(job_id, dataset_id, raw_dir, params)
            return

        # --- Stage 1: Split TIFF into BF + GFP per-frame TIFFs ---
        _set_job(job_id, step="Splitting channels (BF + GFP)...")
        from src.biohack.split_tiffs import split_two_channel_time_stacks

        separated_dir = ds_dir / "separated_frames"
        separated_dir.mkdir(parents=True, exist_ok=True)

        split_two_channel_time_stacks(
            input_file=str(orig_tiff),
            output_dir=str(separated_dir),
            verbose=False,
        )

        # Find the UUID subdir created by split_two_channel_time_stacks
        subdirs = [d for d in separated_dir.iterdir() if d.is_dir() and (d / "brightfield").is_dir()]
        if not subdirs:
            raise RuntimeError("split_two_channel_time_stacks did not produce expected output")
        split_root = subdirs[0]
        bf_dir = split_root / "brightfield"
        gfp_dir = split_root / "gfp"

        # --- Stage 2: Run filament detection on original TIFF (GFP channel) ---
        _set_job(job_id, step="Detecting filaments (Frangi filter)...")
        from src.biohack.experiment_config import ExperimentConfig
        from src.biohack.image_detection import process_time_series_image

        # Set up a run directory under results/
        run_id = str(uuid.uuid4())
        results_base = ds_dir / "results"
        results_base.mkdir(parents=True, exist_ok=True)
        run_dir = results_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Copy split BF+GFP into run dir layout expected by run_filament_pipeline
        import shutil as _shutil
        dst_bf = run_dir / "brightfield"
        dst_gfp = run_dir / "gfp"
        _shutil.copytree(str(bf_dir), str(dst_bf))
        _shutil.copytree(str(gfp_dir), str(dst_gfp))

        # Build config
        base_params = {
            "run_name": dataset_id,
            "dataset_directory": str(split_root),
            "results_directory": str(results_base),
            "use_gpu": False,
            "verbose": False,
        }
        if params:
            base_params.update(params)
        cfg = ExperimentConfig.from_dict(base_params)

        # Run Frangi detection — writes masks into run_dir/filament_mask/
        process_time_series_image(
            image_path=str(orig_tiff),
            output_dir=str(run_dir),
            config=cfg,
            channel=1,  # GFP
        )

        # --- Stage 3: Run cell tracking + filament statistics ---
        _set_job(job_id, step="Running Cellpose segmentation & tracking...")
        from src.biohack.filament_statistics import run_filament_pipeline

        pipeline_result = run_filament_pipeline(cfg, run_id, verbose=False)

        # --- Export for frontend ---
        _set_job(job_id, step="Exporting results for frontend...")
        _export_stats_for_frontend(dataset_id, ds_dir, run_dir, pipeline_result)

        _set_job(job_id, status="complete", dataset_id=dataset_id, step="Done")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _set_job(job_id, status="error", error=str(e))


def _run_legacy_pipeline(
    job_id: str,
    dataset_id: str,
    raw_dir: Path,
    params: dict[str, Any],
) -> None:
    """Fallback: runs old-style detection on pre-split PNGs (no original TIFF available)."""
    from src.biohack.experiment_config import ExperimentConfig
    from src.biohack.image_detection import process_directory

    ds_dir = DATA_DIR / dataset_id
    pipeline_output = ds_dir / "pipeline_output"
    pipeline_output.mkdir(parents=True, exist_ok=True)

    base_params = {
        "run_name": dataset_id,
        "dataset_directory": str(raw_dir),
        "results_directory": str(pipeline_output),
        "use_gpu": False,
        "verbose": False,
    }
    if params:
        base_params.update(params)
    cfg = ExperimentConfig.from_dict(base_params)

    _set_job(job_id, step="Running detection pipeline (legacy)...")

    batch = process_directory(
        cfg,
        max_workers=min(4, os.cpu_count() or 1),
        verbose=False,
    )

    _set_job(job_id, step="Exporting results for frontend...")

    run_dir = batch["run_dir"]
    _export_run_for_frontend(dataset_id, run_dir, batch)

    _set_job(job_id, status="complete", dataset_id=dataset_id, step="Done")


def _export_run_for_frontend(
    dataset_id: str,
    run_dir: Path,
    batch: dict[str, Any],
) -> None:
    """
    Convert pipeline output into the directory structure the frontend expects:
      /data/<dataset_id>/raw/frame_NNN.png
      /data/<dataset_id>/masks/frame_NNN.png
      /data/<dataset_id>/diagnostics/frame_NNN.png
      /data/<dataset_id>/summary.json
    """
    from skimage import io as skio

    ds_dir = DATA_DIR / dataset_id
    raw_out = ds_dir / "raw"
    mask_out = ds_dir / "masks"
    diag_out = ds_dir / "diagnostics"
    raw_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    diag_out.mkdir(parents=True, exist_ok=True)

    masks_src = run_dir / "masks"
    diag_src = run_dir / "images"

    # Build sorted mapping: original stem → frame index
    results = batch.get("results", {})
    sorted_names = sorted(results.keys())

    frame_stats = []

    for idx, name in enumerate(sorted_names):
        stem = Path(name).stem
        r = results[name]

        # Mask → composite (raw + red overlay where mask==1)
        mask_file = masks_src / f"{stem}_mask.png"
        raw_file = raw_out / f"frame_{idx:03d}.png"
        if mask_file.exists() and raw_file.exists():
            raw_img = skio.imread(str(raw_file))
            mask_img = skio.imread(str(mask_file))
            # Build RGBA composite: raw as grayscale base, red where mask > 0
            if raw_img.ndim == 2:
                rgba = np.stack([raw_img, raw_img, raw_img,
                                 np.full_like(raw_img, 255)], axis=-1)
            else:
                rgba = np.dstack([raw_img[:, :, :3],
                                  np.full(raw_img.shape[:2], 255, dtype=np.uint8)])
            mask_bool = mask_img > 0
            if mask_bool.ndim > 2:
                mask_bool = mask_bool.any(axis=-1)
            rgba[mask_bool, 0] = np.clip(
                rgba[mask_bool, 0].astype(np.int16) + 140, 0, 255).astype(np.uint8)
            rgba[mask_bool, 1] = (rgba[mask_bool, 1] * 0.3).astype(np.uint8)
            rgba[mask_bool, 2] = (rgba[mask_bool, 2] * 0.3).astype(np.uint8)
            rgba[mask_bool, 3] = 255
            skio.imsave(str(mask_out / f"frame_{idx:03d}.png"), rgba,
                        check_contrast=False)
        elif mask_file.exists():
            import shutil
            shutil.copy2(str(mask_file), str(mask_out / f"frame_{idx:03d}.png"))

        # Diagnostic figure → PNG
        diag_file = diag_src / f"{stem}_pipeline.png"
        if diag_file.exists():
            import shutil
            shutil.copy2(str(diag_file), str(diag_out / f"frame_{idx:03d}.png"))

        stats = r.get("stats", {})
        frame_stats.append({
            "frame": idx,
            "original_name": name,
            "filament_present": r.get("filament_present", False),
            "foreground_pixels": stats.get("foreground_pixels", 0),
            "num_components": stats.get("num_components", 0),
        })

    # Summary JSON
    from datetime import datetime, timezone
    summary = {
        "dataset_id": dataset_id,
        "run_uid": batch.get("run_uid", ""),
        "run_name": "",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "frame_count": len(sorted_names),
        "frames_with_filament": sum(1 for f in frame_stats if f["filament_present"]),
        "frame_stats": frame_stats,
    }
    with open(ds_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Copy any CSV files from the pipeline run directory
    import shutil as _shutil
    for csv_file in run_dir.glob("*.csv"):
        _shutil.copy2(str(csv_file), str(ds_dir / csv_file.name))
    # Also check pipeline_output root for CSVs
    po_dir = ds_dir / "pipeline_output"
    if po_dir.exists():
        for csv_file in po_dir.glob("*.csv"):
            _shutil.copy2(str(csv_file), str(ds_dir / csv_file.name))

    # Ensure cell_tracks.csv exists — use sample data until pipeline produces real output
    cell_tracks_path = ds_dir / "cell_tracks.csv"
    if not cell_tracks_path.exists() and SAMPLE_CSV.exists():
        _shutil.copy2(str(SAMPLE_CSV), str(cell_tracks_path))

    # Update datasets.json
    datasets = _read_datasets()
    existing_ids = {d["id"] for d in datasets}
    if dataset_id not in existing_ids:
        datasets.append({
            "id": dataset_id,
            "name": dataset_id.replace("_", " ").title(),
            "frames": len(sorted_names),
            "timeSpan": _format_timespan(len(sorted_names)),
        })
        _write_datasets(datasets)


def _export_stats_for_frontend(
    dataset_id: str,
    ds_dir: Path,
    filament_output: Path,
    pipeline_result: Any,
) -> None:
    """
    Export the new-pipeline results into the frontend-expected structure.

    Uses filament detection masks for mask overlay PNGs, and copies
    the cell_tracks_per_frame_with_filaments_and_ids.csv as cell_tracks.csv.
    """
    import shutil as _shutil
    from datetime import datetime, timezone
    from skimage import io as skio

    raw_out = ds_dir / "raw"
    mask_out = ds_dir / "masks"
    diag_out = ds_dir / "diagnostics"
    mask_out.mkdir(parents=True, exist_ok=True)
    diag_out.mkdir(parents=True, exist_ok=True)

    # Count frames from raw PNGs
    raw_frames = sorted(raw_out.glob("frame_*.png"))
    n_frames = len(raw_frames)

    # Build mask overlay PNGs from filament detection output
    masks_src = filament_output / "filament_mask"
    diag_src = filament_output / "diagnostics"

    if masks_src.is_dir():
        mask_files = sorted(masks_src.glob("*_mask_*"))
        for mf in mask_files:
            # Parse frame index from filename: {stem}_mask_{t}_{channel}.png
            parts = mf.stem.rsplit("_mask_", 1)
            if len(parts) < 2:
                continue
            idx_parts = parts[1].split("_")
            try:
                frame_idx = int(idx_parts[0])
            except ValueError:
                continue

            raw_file = raw_out / f"frame_{frame_idx:03d}.png"
            out_file = mask_out / f"frame_{frame_idx:03d}.png"

            if raw_file.exists():
                raw_img = skio.imread(str(raw_file))
                mask_img = skio.imread(str(mf))
                # Build composite: raw base + red overlay where mask > 0
                if raw_img.ndim == 2:
                    rgba = np.stack([raw_img, raw_img, raw_img,
                                     np.full_like(raw_img, 255)], axis=-1)
                else:
                    rgba = np.dstack([raw_img[:, :, :3],
                                      np.full(raw_img.shape[:2], 255, dtype=np.uint8)])
                mask_bool = mask_img > 0
                if mask_bool.ndim > 2:
                    mask_bool = mask_bool.any(axis=-1)
                rgba[mask_bool, 0] = np.clip(
                    rgba[mask_bool, 0].astype(np.int16) + 140, 0, 255).astype(np.uint8)
                rgba[mask_bool, 1] = (rgba[mask_bool, 1] * 0.3).astype(np.uint8)
                rgba[mask_bool, 2] = (rgba[mask_bool, 2] * 0.3).astype(np.uint8)
                rgba[mask_bool, 3] = 255
                skio.imsave(str(out_file), rgba, check_contrast=False)
            else:
                _shutil.copy2(str(mf), str(out_file))

    # Copy diagnostic images
    if diag_src.is_dir():
        diag_files = sorted(diag_src.glob("*_pipeline_*"))
        for df in diag_files:
            parts = df.stem.rsplit("_pipeline_", 1)
            if len(parts) < 2:
                continue
            idx_parts = parts[1].split("_")
            try:
                frame_idx = int(idx_parts[0])
            except ValueError:
                continue
            _shutil.copy2(str(df), str(diag_out / f"frame_{frame_idx:03d}.png"))

    # Copy statistics CSVs to dataset root
    stats_dir = filament_output / "statistics"
    if stats_dir.is_dir():
        for csv_file in stats_dir.glob("*.csv"):
            _shutil.copy2(str(csv_file), str(ds_dir / csv_file.name))

    # Copy the main tracking CSV as cell_tracks.csv (what the frontend loads)
    main_csv = ds_dir / "cell_tracks_per_frame_with_filaments_and_ids.csv"
    cell_tracks_dst = ds_dir / "cell_tracks.csv"
    if main_csv.exists():
        _shutil.copy2(str(main_csv), str(cell_tracks_dst))

    # Build summary JSON
    df = pipeline_result.cell_tracks_with_filaments_and_ids_df
    frames_with_filament = 0
    if "filament_present" in df.columns:
        frames_with_filament = int(df.groupby("frame")["filament_present"].max().sum())

    summary = {
        "dataset_id": dataset_id,
        "run_uid": "",
        "run_name": "",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "frame_count": n_frames,
        "frames_with_filament": frames_with_filament,
        "frame_stats": [],
    }
    with open(ds_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload a multi-frame TIFF → split frames → create dataset → run pipeline."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    fname = secure_filename(file.filename)
    ext = os.path.splitext(fname)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Only TIFF files accepted, got {ext}"}), 400

    # Save upload
    UPLOAD_TMP.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_TMP / f"{uuid.uuid4().hex}_{fname}"
    file.save(str(upload_path))

    # Create dataset
    dataset_id = Path(fname).stem.lower().replace(" ", "_")
    # Reject duplicate by name
    display_name = Path(fname).stem.replace("_", " ").title()
    existing_datasets = _read_datasets()
    existing_names = {d.get("name", "").lower() for d in existing_datasets}
    if display_name.lower() in existing_names:
        upload_path.unlink(missing_ok=True)
        return jsonify({"error": f"A dataset named \"{display_name}\" already exists"}), 409
    # Avoid id collisions
    existing_ids = {d["id"] for d in existing_datasets}
    if dataset_id in existing_ids:
        dataset_id = f"{dataset_id}_{uuid.uuid4().hex[:6]}"

    ds_dir = DATA_DIR / dataset_id
    raw_dir = ds_dir / "raw"

    job_id = uuid.uuid4().hex

    def _upload_job():
        try:
            _set_job(job_id, status="running", step="Splitting TIFF into frames...")
            n_frames = _split_tiff_to_pngs(upload_path, raw_dir, dataset_id)

            # Register dataset
            datasets = _read_datasets()
            datasets.append({
                "id": dataset_id,
                "name": Path(fname).stem.replace("_", " ").title(),
                "frames": n_frames,
                "timeSpan": _format_timespan(n_frames),
            })
            _write_datasets(datasets)

            _set_job(job_id, status="complete", dataset_id=dataset_id, step="Done")

        except Exception as e:
            _set_job(job_id, status="error", error=str(e))
        finally:
            # Cleanup upload
            upload_path.unlink(missing_ok=True)

    _set_job(job_id, status="running", dataset_id=dataset_id, step="Uploading...")
    t = threading.Thread(target=_upload_job, daemon=True)
    t.start()

    return jsonify({"job_id": job_id}), 202


@app.route("/api/run", methods=["POST"])
def api_run():
    """Run (or re-run) the pipeline on an existing dataset with custom params."""
    body = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    params = body.get("params", {})

    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400

    ds_dir = DATA_DIR / dataset_id
    raw_dir = ds_dir / "raw"
    if not raw_dir.is_dir():
        return jsonify({"error": f"No raw frames found for dataset '{dataset_id}'"}), 404

    job_id = uuid.uuid4().hex
    _set_job(job_id, status="running", dataset_id=dataset_id, step="Queued...")

    t = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, dataset_id, raw_dir, params),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id}), 202


@app.route("/api/jobs")
def api_jobs():
    """Return all currently-running jobs so new devices can discover them."""
    with _jobs_lock:
        active = {
            jid: j for jid, j in _jobs.items()
            if j.get("status") not in ("complete", "error")
        }
    return jsonify(active)


@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/api/datasets")
def api_datasets():
    datasets = _read_datasets()
    for d in datasets:
        ds_dir = DATA_DIR / d["id"]
        summary_path = ds_dir / "summary.json"
        d["has_results"] = summary_path.exists()
        if summary_path.exists():
            with open(summary_path) as f:
                summ = json.load(f)
            d["run_name"] = summ.get("run_name", "")
            d["completed_at"] = summ.get("completed_at", "")
            # Ensure cell_tracks.csv is always present for datasets with results
            cell_tracks_path = ds_dir / "cell_tracks.csv"
            if not cell_tracks_path.exists() and SAMPLE_CSV.exists():
                import shutil
                shutil.copy2(str(SAMPLE_CSV), str(cell_tracks_path))
    return jsonify(datasets)


@app.route("/api/dataset/<dataset_id>", methods=["DELETE"])
def api_delete_dataset(dataset_id: str):
    """Delete a dataset and all its files."""
    import shutil

    ds_dir = DATA_DIR / dataset_id
    if ds_dir.is_dir():
        shutil.rmtree(ds_dir)

    # Also clean up the stored original TIFF
    orig_dir = ORIG_TIFFS / dataset_id
    if orig_dir.is_dir():
        shutil.rmtree(orig_dir)

    datasets = _read_datasets()
    datasets = [d for d in datasets if d["id"] != dataset_id]
    _write_datasets(datasets)

    return jsonify({"ok": True})


@app.route("/api/dataset/<dataset_id>/run-name", methods=["PATCH"])
def api_update_run_name(dataset_id: str):
    """Update the run_name in summary.json."""
    body = request.get_json(silent=True) or {}
    run_name = body.get("run_name", "")
    ds_dir = DATA_DIR / dataset_id
    summary_path = ds_dir / "summary.json"
    if not summary_path.exists():
        return jsonify({"error": "No results found"}), 404
    with open(summary_path) as f:
        summ = json.load(f)
    summ["run_name"] = run_name
    with open(summary_path, "w") as f:
        json.dump(summ, f, indent=2)
    return jsonify({"ok": True})


@app.route("/api/dataset/<dataset_id>/results", methods=["DELETE"])
def api_delete_results(dataset_id: str):
    """Delete only pipeline results (masks, diagnostics, summary, pipeline_output) but keep raw frames."""
    import shutil

    ds_dir = DATA_DIR / dataset_id
    for sub in ("masks", "diagnostics", "pipeline_output", "separated_frames",
                "filament_detection", "results", "cellpose_masks"):
        p = ds_dir / sub
        if p.is_dir():
            shutil.rmtree(p)
    for csv_file in ds_dir.glob("*.csv"):
        csv_file.unlink()
    summary = ds_dir / "summary.json"
    if summary.exists():
        summary.unlink()

    return jsonify({"ok": True})




@app.route("/api/dataset/<dataset_id>/download", methods=["POST"])
def api_download(dataset_id: str):
    """Build a dataset zip and return it as a download attachment."""
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "results")
    ds_dir = DATA_DIR / dataset_id
    if not ds_dir.is_dir():
        return jsonify({"error": "Dataset not found"}), 404
    if mode not in {"results", "all"}:
        return jsonify({"error": f"Unsupported download mode '{mode}'"}), 400

    screenshots = body.get("screenshots", [])
    # Limit screenshot count and size to prevent abuse
    if not isinstance(screenshots, list):
        screenshots = []
    screenshots = screenshots[:20]  # max 20 images

    fd, tmp_zip = tempfile.mkstemp(prefix=f"{dataset_id}_{mode}_", suffix=".zip")
    os.close(fd)
    zip_path = Path(tmp_zip)

    try:
        _write_dataset_zip(zip_path, ds_dir, mode, screenshots=screenshots)
    except Exception:
        zip_path.unlink(missing_ok=True)
        raise

    @after_this_request
    def _cleanup_download(response):
        zip_path.unlink(missing_ok=True)
        return response

    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{dataset_id}_{mode}.zip",
        max_age=0,
    )


# ---------------------------------------------------------------------------
# Stripe checkout
# ---------------------------------------------------------------------------

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")


@app.route("/api/checkout", methods=["POST"])
def api_checkout():
    """Create a Stripe Checkout session and return the redirect URL."""
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        origin = request.headers.get("Origin", request.host_url.rstrip("/"))
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=f"{origin}/?checkout=success",
            cancel_url=f"{origin}/?checkout=cancelled",
        )
        return jsonify({"url": session.url})
    except ImportError:
        return jsonify({"error": "Stripe not installed on server"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Static file serving for /data/
# ---------------------------------------------------------------------------

@app.route("/data/<path:filepath>")
def serve_data(filepath: str):
    return send_from_directory(str(DATA_DIR), filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=False)
