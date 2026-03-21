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
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # biohack/
DATA_DIR = BASE_DIR / "server_data"                        # served as /data/
UPLOAD_TMP = BASE_DIR / "server_uploads"
ALLOWED_EXTENSIONS = {".tif", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024      # 500 MB

# In-memory job tracker  {job_id: {status, step, error, dataset_id}}
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _split_tiff_to_pngs(tiff_path: Path, out_dir: Path) -> int:
    """Split a multi-frame TIFF into individual PNG frames. Returns frame count."""
    import tifffile
    from skimage import exposure, io as skio

    stack = tifffile.imread(str(tiff_path))
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]  # single frame

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(stack):
        # Normalize to uint8 for browser display
        if frame.dtype != np.uint8:
            frame_f = frame.astype(np.float64)
            lo, hi = np.percentile(frame_f, [0.5, 99.5])
            if hi - lo < 1e-9:
                hi = lo + 1.0
            frame_f = np.clip((frame_f - lo) / (hi - lo), 0, 1)
            frame_8 = (frame_f * 255).astype(np.uint8)
        else:
            frame_8 = frame
        skio.imsave(str(out_dir / f"frame_{i:03d}.png"), frame_8, check_contrast=False)
    return len(stack)


def _run_pipeline_job(
    job_id: str,
    dataset_id: str,
    raw_dir: Path,
    params: dict[str, Any],
) -> None:
    """Background worker: runs the pipeline, exports results, updates job status."""
    import sys
    sys.path.insert(0, str(BASE_DIR))

    try:
        from src.biohack.image_detection import FilamentConfig, filament_config_from_mapping, process_directory

        _set_job(job_id, status="running", step="Initializing pipeline...")

        ds_dir = DATA_DIR / dataset_id
        pipeline_output = ds_dir / "pipeline_output"
        pipeline_output.mkdir(parents=True, exist_ok=True)

        # Build config from params
        cfg = filament_config_from_mapping(params) if params else FilamentConfig()

        _set_job(job_id, step="Running detection pipeline...")

        batch = process_directory(
            input_dir=raw_dir,
            output_dir=pipeline_output,
            config=cfg,
            max_workers=min(4, os.cpu_count() or 1),
            verbose=False,
        )

        _set_job(job_id, step="Exporting results for frontend...")

        run_dir = batch["run_dir"]
        _export_run_for_frontend(dataset_id, run_dir, batch)

        _set_job(job_id, status="complete", dataset_id=dataset_id, step="Done")

    except Exception as e:
        _set_job(job_id, status="error", error=str(e))


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

    # Update datasets.json
    datasets = _read_datasets()
    existing_ids = {d["id"] for d in datasets}
    if dataset_id not in existing_ids:
        datasets.append({
            "id": dataset_id,
            "name": dataset_id.replace("_", " ").title(),
            "frames": len(sorted_names),
            "timeSpan": f"{len(sorted_names)} frames",
        })
        _write_datasets(datasets)


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
    # Avoid collisions
    existing = {d["id"] for d in _read_datasets()}
    if dataset_id in existing:
        dataset_id = f"{dataset_id}_{uuid.uuid4().hex[:6]}"

    ds_dir = DATA_DIR / dataset_id
    raw_dir = ds_dir / "raw"

    job_id = uuid.uuid4().hex

    def _upload_job():
        try:
            _set_job(job_id, status="running", step="Splitting TIFF into frames...")
            n_frames = _split_tiff_to_pngs(upload_path, raw_dir)

            # Register dataset
            datasets = _read_datasets()
            datasets.append({
                "id": dataset_id,
                "name": Path(fname).stem.replace("_", " ").title(),
                "frames": n_frames,
                "timeSpan": f"{n_frames} frames",
            })
            _write_datasets(datasets)

            _set_job(job_id, status="complete", dataset_id=dataset_id, step="Done")

        except Exception as e:
            _set_job(job_id, status="error", error=str(e))
        finally:
            # Cleanup upload
            upload_path.unlink(missing_ok=True)

    _set_job(job_id, status="running", step="Uploading...")
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
    _set_job(job_id, status="running", step="Queued...")

    t = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, dataset_id, raw_dir, params),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id}), 202


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
    return jsonify(datasets)


@app.route("/api/dataset/<dataset_id>", methods=["DELETE"])
def api_delete_dataset(dataset_id: str):
    """Delete a dataset and all its files."""
    import shutil

    ds_dir = DATA_DIR / dataset_id
    if ds_dir.is_dir():
        shutil.rmtree(ds_dir)

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
    for sub in ("masks", "diagnostics", "pipeline_output"):
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
    """Build a zip and place it in the dataset dir for static serving."""
    import zipfile

    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "results")
    ds_dir = DATA_DIR / dataset_id
    if not ds_dir.is_dir():
        return jsonify({"error": "Dataset not found"}), 404

    zip_name = f"{dataset_id}_{mode}.zip"
    zip_path = ds_dir / zip_name

    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
        for csv_f in sorted(ds_dir.glob("*.csv")):
            zf.write(csv_f, csv_f.name)
        summary_p = ds_dir / "summary.json"
        if summary_p.exists():
            zf.write(summary_p, "summary.json")

        if mode == "all":
            for subdir in ("raw", "masks", "diagnostics"):
                sub = ds_dir / subdir
                if sub.is_dir():
                    for fp in sorted(sub.iterdir()):
                        if fp.is_file():
                            zf.write(fp, f"{subdir}/{fp.name}")

    return jsonify({"url": f"/data/{dataset_id}/{zip_name}"})


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
