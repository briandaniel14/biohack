# Filament Detection Tool — System Architecture

## 1. Overview

A web-based microscopy analysis tool that detects and tracks filamentous structures in time-lapse fluorescence microscopy images. Users upload grayscale TIF stacks via a browser UI, the server runs an 8-stage detection pipeline, and results are visualised in a React frontend with playback, per-track inspection, and metrics dashboards.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User's Browser                              │
│                                                                     │
│  ┌─────────────────────┐       ┌──────────────────────────────────┐ │
│  │     Process Page     │       │          Results Page            │ │
│  │  Upload TIF          │       │  Frame playback + controls      │ │
│  │  Adjust hyperparams  │       │  Track inspector table          │ │
│  │  Preview frames      │       │  Metrics dashboard (4 charts)   │ │
│  │  Run pipeline        │       │  Click-to-select tracks         │ │
│  └──────────┬──────────┘       └──────────────┬───────────────────┘ │
│             │                                  │                     │
│             │  POST /api/upload                │  GET /data/...      │
│             │  GET  /api/status/:id            │  (static files)     │
│             │  GET  /api/datasets              │                     │
└─────────────┼──────────────────────────────────┼─────────────────────┘
              │                                  │
              ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EC2 Instance (35.179.135.58)                      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      nginx (port 80)                         │   │
│  │                                                              │   │
│  │  /           → serve /home/ubuntu/frontend/ (SPA fallback)   │   │
│  │  /data/      → serve /home/ubuntu/frontend/data/ (static)    │   │
│  │  /api/*      → proxy to 127.0.0.1:8080 (Flask)              │   │
│  │  *.js,css,…  → 1-day cache, immutable                       │   │
│  │  client_max_body_size = 500M                                 │   │
│  └─────────────────────────┬────────────────────────────────────┘   │
│                            │                                        │
│  ┌─────────────────────────▼────────────────────────────────────┐   │
│  │         Flask API — upload_server.py (port 8080)             │   │
│  │         systemd service: upload-api                          │   │
│  │         Python venv: /home/ubuntu/venv                       │   │
│  │                                                              │   │
│  │  POST /api/upload    → save TIF → spawn pipeline thread      │   │
│  │  GET  /api/status/id → return job progress from memory       │   │
│  │  GET  /api/datasets  → return datasets.json                  │   │
│  └─────────────────────────┬────────────────────────────────────┘   │
│                            │                                        │
│                   (background thread)                               │
│                            │                                        │
│  ┌─────────────────────────▼────────────────────────────────────┐   │
│  │         filament_detection.run_pipeline()                    │   │
│  │         → subprocess with 600s timeout                       │   │
│  └─────────────────────────┬────────────────────────────────────┘   │
│                            │                                        │
│  ┌─────────────────────────▼────────────────────────────────────┐   │
│  │         export_for_frontend.export_dataset()                 │   │
│  │         → subprocess with 300s timeout                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Infrastructure

### EC2 Instance

| Property | Value |
|---|---|
| Host | `35.179.135.58` |
| OS | Ubuntu |
| SSH key | `~/.ssh/HomeMBP.pem` |
| User | `ubuntu` |
| Python venv | `/home/ubuntu/venv` |

### Server Directory Layout

```
/home/ubuntu/
├── frontend/                  # Built React app (rsync'd from dev machine)
│   ├── index.html
│   ├── assets/                # Vite-bundled JS/CSS
│   └── data/                  # Dataset assets served as static files
│       ├── datasets.json      # Dataset index
│       └── dataset_NN/
│           ├── raw/           # frame_000.png … frame_099.png
│           ├── annotated/     # frame_000.png … frame_099.png (with overlays)
│           ├── structure_measurements.csv
│           └── track_summary.csv
├── pipeline/                  # Pipeline code (filament_detection.py, export_for_frontend.py, upload_server.py)
├── pipeline_output/           # Raw pipeline outputs per dataset
│   └── <tif_stem>/
│       ├── <stem>_filament_measurements.csv
│       ├── <stem>_track_summary.csv
│       └── <stem>_annotated.tif
└── uploads/                   # Uploaded TIF files
```

### nginx Configuration

```nginx
server {
    listen 80;
    server_name _;
    root /home/ubuntu/frontend;
    index index.html;
    client_max_body_size 500M;

    location /api/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }

    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
```

Config file: `/etc/nginx/sites-enabled/frontend`

### systemd Service

```ini
[Unit]
Description=Filament Upload API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/pipeline
ExecStart=/home/ubuntu/venv/bin/python3 upload_server.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Service name: `upload-api`
Manage: `sudo systemctl {start|stop|restart|status} upload-api`
Logs: `journalctl -u upload-api -f`

## 3. Backend — Flask API

**File:** `test/upload_server.py` (deployed to `/home/ubuntu/pipeline/upload_server.py`)
**Port:** 8080 (bound to `0.0.0.0`)

### Endpoints

#### `POST /api/upload`

Upload a TIF stack for processing.

**Request:** `multipart/form-data` with field `file`

**Validation:**
- File must be present and have a filename
- Extension must be `.tif` or `.tiff`
- File must be a valid TIFF image stack (verified with `tifffile`)

**Response (202):**
```json
{ "job_id": "a1b2c3d4", "filename": "sample.tif" }
```

**Error (400):**
```json
{ "error": "No file part" | "No selected file" | "Only .tif/.tiff files accepted" | "Not a valid TIFF image" }
```

**Side effects:**
- Saves file to `/home/ubuntu/uploads/<safe_name>`
- Starts background processing thread (see §4)

#### `GET /api/status/<job_id>`

Poll processing status.

**Response (200):**
```json
{
  "status": "queued" | "processing" | "complete" | "error",
  "step": "Running detection pipeline...",
  "dataset_id": "dataset_03",
  "error": "Pipeline failed: ..."
}
```

Fields `step`, `dataset_id`, and `error` are present only when applicable.

**Note:** Job state is held **in-memory** (Python dict). All job state is lost on server restart.

#### `GET /api/datasets`

Returns the current dataset index.

**Response (200):**
```json
[
  { "id": "dataset_01", "name": "ch20_URA7_URA8_001-crop1", "frames": 100, "timeSpan": "24h", "frameInterval": 14.4 }
]
```

#### `POST /api/dataset/<dataset_id>/download`

Builds a ZIP archive for the selected dataset and returns it directly as an attachment response.

**Request:**
```json
{ "mode": "results" | "all" }
```

**Behavior:**
- `results` includes `summary.json` plus dataset CSV outputs
- `all` also includes `raw/`, `masks/`, and `diagnostics/` image folders
- The ZIP is generated in a temporary file and streamed back in the same request
- The frontend downloads the response body as a blob instead of opening a static `/data/...zip` URL in a new tab

Reads from `/home/ubuntu/frontend/data/datasets.json`.

### Background Processing Thread

When a file is uploaded, a daemon thread runs the following steps sequentially:

```
1. Update job status → "processing", step "Running detection pipeline..."
2. subprocess.run() → filament_detection.run_pipeline(tif_path, output_dir, Config())
   - Timeout: 600 seconds
   - Runs in a separate Python process (imports from /home/ubuntu/pipeline/)
3. Update job status → step "Exporting for frontend..."
4. subprocess.run() → export_for_frontend.export_dataset(tif_path, output_dir, frontend_dir, dataset_id)
   - Timeout: 300 seconds
5. Fix file permissions: chmod -R o+rX on the frontend data directory
6. Update datasets.json: append new entry with auto-incremented dataset_NN ID
7. Update job status → "complete", with dataset_id
```

On any failure: job status → `"error"` with the exception message.

**Dataset ID generation:** scans existing `dataset_*` directories under `/home/ubuntu/frontend/data/`, finds the highest number, increments by 1.

## 4. Detection Pipeline

**File:** `test/filament_detection.py` (deployed to `/home/ubuntu/pipeline/filament_detection.py`)

### Input

- 100-frame grayscale TIF stack (128×128 px, uint16)
- ~14.4 minutes between frames, covering a 24-hour time-lapse

### Pipeline Stages

```
TIF Stack
  │
  ▼
┌─────────────────────┐
│ 1. Preprocessing     │  Load → min-max normalise per frame → Gaussian denoise
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. Hessian Tubeness  │  Multi-scale Hessian eigenvalue analysis
│                      │  Tubeness = |λ₂| where λ₂ < 0, max across scales
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. LoG Blob Detect   │  skimage blob_log for round structures
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. Segmentation      │  Threshold tubeness (percentile/Otsu) + blob masks
│                      │  Morphological closing → remove small objects → label
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. Classification    │  Three-gate filter per region:
│    & Measurement     │    Gate 1: area ≥ min_area
│                      │    Gate 2: cell exclusion (area + eccentricity)
│                      │    Gate 3: intensity ≥ 2× background mean
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 6. Tracking          │  Greedy IoU matching → centroid fallback → new tracks
│                      │  Gap tolerance, min track length filter
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 7. Output Metrics    │  structure_measurements.csv (per-frame)
│                      │  track_summary.csv (per-track)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 8. Visualisation     │  Annotated RGB TIF with colour bounding boxes
│                      │  Active structures plot, lifetime histogram
└─────────────────────┘
```

### Config Dataclass (17 parameters)

| Parameter | Default | Purpose |
|---|---|---|
| `gaussian_sigma` | 1.0 | Gaussian blur sigma for denoising |
| `hessian_sigmas` | [1.5, 2.0, 3.0, 4.0, 5.0] | Scales for Hessian tubeness |
| `blob_min_sigma` | 2.0 | LoG blob detector min sigma |
| `blob_max_sigma` | 6.0 | LoG blob detector max sigma |
| `blob_threshold` | 0.02 | LoG blob detection threshold |
| `threshold_method` | `"percentile"` | Segmentation method (`"otsu"` or `"percentile"`) |
| `threshold_percentile` | 90 | Percentile for segmentation threshold |
| `min_area` | 30 | Minimum object area (pixels) |
| `closing_radius` | 1 | Morphological closing disk radius |
| `cell_min_area` | 200 | Cell exclusion: min area to flag as cell |
| `cell_max_eccentricity` | 0.6 | Cell exclusion: max eccentricity |
| `max_area` | 350 | Hard cap — anything larger is a cell |
| `intensity_threshold` | 2.0 | Structure/background intensity ratio gate |
| `iou_threshold` | 0.3 | Tracking IoU threshold |
| `max_centroid_dist` | 15.0 | Tracking centroid distance fallback |
| `gap_frames` | 1 | Tracking gap tolerance (frames) |
| `min_track_length` | 2 | Minimum track duration to keep |
| `frame_interval_min` | 14.4 | Minutes per frame (for time conversion) |

### Pipeline Outputs

| File | Description |
|---|---|
| `<stem>_filament_measurements.csv` | Per-frame per-structure rows with geometry, intensity, morphology |
| `<stem>_track_summary.csv` | Per-track aggregated stats |
| `<stem>_annotated.tif` | RGB TIF with colour-coded bounding boxes |
| `<stem>_active_structures.png` | Active structure count over time |
| `<stem>_lifetime_hist.png` | Track lifetime histogram |
| `length_over_time.png` | Filament length over time |

### Invocation

```python
from filament_detection import run_pipeline, Config

cfg = Config(gaussian_sigma=1.5, min_area=50)
measurements_df, track_summary_df = run_pipeline("input.tif", "output/", cfg)
```

CLI: `python filament_detection.py <data_dir> [-o output_dir]`

## 5. Export Script

**File:** `test/export_for_frontend.py` (deployed to `/home/ubuntu/pipeline/export_for_frontend.py`)

Converts raw pipeline outputs into the directory structure the frontend expects.

### Processing Steps

```
Pipeline Output                          Frontend Data
─────────────────                        ──────────────
source.tif                    ──►  dataset_NN/raw/frame_000.png … frame_099.png
                                   (normalised to 0-255, grayscale PNG)

<stem>_annotated.tif          ──►  dataset_NN/annotated/frame_000.png … frame_099.png
                                   (RGB PNG per frame)

<stem>_filament_measurements  ──►  dataset_NN/structure_measurements.csv
                                   (column renames: major_axis_length → major_axis, etc.)

<stem>_track_summary.csv      ──►  dataset_NN/track_summary.csv
                                   (+ computed morphology fields)
```

### Column Renames (Measurements)

| Pipeline Column | Frontend Column |
|---|---|
| `major_axis_length` | `major_axis` |
| `minor_axis_length` | `minor_axis` |
| `orientation_deg` | `orientation` |
| `area_px` | `area` |

### Computed Fields (Track Summary)

| Field | Derivation |
|---|---|
| `initial_morphology` | Morphology at formation frame |
| `final_morphology` | Morphology at dissolution frame |
| `morphology_label` | `"filament"`, `"condensate"`, or `"mixed"` |
| `num_transitions` | Count of morphology changes across track lifetime |

## 6. Frontend

**Stack:** React 19, Vite 6, Tailwind CSS 3, Recharts, PapaParse
**Source:** `test/frontend/`
**Build:** `npx vite build` → `dist/`
**Deploy:** `rsync -avz --delete dist/ ubuntu@35.179.135.58:/home/ubuntu/frontend/ -e "ssh -i ~/.ssh/HomeMBP.pem"`

### Two-Page Architecture

```
App.jsx (shell)
├── Header: logo + title + nav tabs [Process | Results]
├── Shared state: datasets[], currentDataset, page
│
├── ProcessPage.jsx
│   ├── Left: upload zone, dataset selector, frame preview + scrubber
│   └── Right: HyperparamPanel + "Run Pipeline" button
│
└── ResultsPage.jsx
    ├── Left (42%): frame player with full playback controls
    └── Right (58%):
        ├── TrackInspector.jsx (45%): sortable track table + detail panel
        └── MetricsDashboard.jsx (55%): 2×2 chart grid
```

### Source Files

| File | Purpose |
|---|---|
| `src/App.jsx` | Shell — shared dataset state, header with nav tabs, page routing |
| `src/ProcessPage.jsx` | Upload, preview, hyperparameter adjustment, pipeline trigger |
| `src/ResultsPage.jsx` | Playback, track selection, inspector, metrics |
| `src/TrackInspector.jsx` | Sortable track table, selected track detail with sparklines |
| `src/MetricsDashboard.jsx` | 2×2 chart grid (length, count, eccentricity, axes) |
| `src/HyperparamPanel.jsx` | Parameter sliders, numeric inputs, portal-rendered tooltips |
| `src/data.js` | Data loading, frame URLs, track colours, upload/poll API calls |
| `src/main.jsx` | React entry point |
| `src/index.css` | Tailwind directives |

### Data Loading (Static Files)

All dataset viewing is done via static file fetches — no backend needed:

- `GET /data/datasets.json` → dataset index
- `GET /data/dataset_NN/structure_measurements.csv` → parsed with PapaParse
- `GET /data/dataset_NN/track_summary.csv` → parsed with PapaParse
- `GET /data/dataset_NN/{raw|annotated}/frame_NNN.png` → frame images

All 200 frames (100 raw + 100 annotated) are preloaded on dataset switch.

### Upload Flow (Current Implementation)

```
User drags .tif onto upload zone
         │
         ▼
ProcessPage validates extension (.tif/.tiff)
         │
         ▼
POST /api/upload (FormData with file)
         │
         ▼
Server saves to /home/ubuntu/uploads/, returns { job_id }
         │
         ▼
ProcessPage polls GET /api/status/{job_id} every 2 seconds
         │
         ├── status: "processing"  → show step text in UI
         ├── status: "error"       → show error, stop polling
         └── status: "complete"    → refresh dataset list,
                                     select new dataset,
                                     enable annotated view
```

## 7. Current Gaps & TODOs

### "Run Pipeline" Button — Not Yet Wired

The "Run Pipeline" button on the Process page is a **placeholder stub**:

```jsx
onClick={() => {/* TODO: POST hyperparams to backend and re-run pipeline */}}
```

To make it functional, the following work is needed:

#### Frontend Changes

1. **HyperparamPanel must expose its values** — currently all parameter state is local to the component via `useState`. The parent (`ProcessPage`) cannot read the values. Options:
   - Lift state up: pass `values` and `onChange` props
   - Use `useImperativeHandle` + `forwardRef` to expose a `getValues()` method
   - Use React Context

2. **ProcessPage must POST hyperparams** — the "Run Pipeline" button should call a new API endpoint with the current hyperparameter values and the selected dataset ID.

3. **Poll for completion** — reuse the existing polling pattern from the upload flow.

#### Backend Changes

1. **New endpoint: `POST /api/run`** — accepts a JSON body with dataset ID and hyperparameters, triggers re-processing of an existing dataset.

   Proposed schema:
   ```json
   POST /api/run
   {
     "dataset_id": "dataset_01",
     "params": {
       "gaussian_sigma": 1.5,
       "min_area": 50,
       ...
     }
   }
   ```
   Response: `{ "job_id": "..." }` — same polling pattern as upload.

2. **Map frontend params to pipeline Config** — the HyperparamPanel exposes 9 parameters from `image_detection.yaml` (Frangi-based). The actual pipeline `Config` has 17 parameters (Hessian-based). These are **different parameter sets** from different pipeline versions. Decision needed:
   - Option A: Update HyperparamPanel to expose the actual `Config` parameters
   - Option B: Update the pipeline to accept the `image_detection.yaml` parameters
   - Option C: Map between the two where possible, ignore the rest

3. **Re-processing logic** — the run endpoint would:
   - Look up the original TIF from the uploads directory (or store a mapping)
   - Run `filament_detection.run_pipeline()` with custom `Config`
   - Run `export_for_frontend.export_dataset()` to overwrite the dataset's frontend files
   - Update job status as complete

### Parameter Mismatch Detail

| HyperparamPanel (from `image_detection.yaml`) | Pipeline Config (from `filament_detection.py`) |
|---|---|
| `clip_low_percentile` | — (not in Config) |
| `clip_high_percentile` | — (not in Config) |
| `gaussian_sigma` | `gaussian_sigma` ✓ |
| `foreground_percentile` | — (not in Config) |
| `local_block_size` | — (not in Config) |
| `local_offset` | — (not in Config) |
| `frangi_threshold_percentile` | — (not in Config; uses Hessian, not Frangi) |
| `min_object_size` | `min_area` (similar purpose) |
| `min_pixels_for_presence` | — (not in Config) |
| — | `hessian_sigmas` (not in panel) |
| — | `blob_min_sigma` (not in panel) |
| — | `blob_max_sigma` (not in panel) |
| — | `blob_threshold` (not in panel) |
| — | `threshold_method` (not in panel) |
| — | `threshold_percentile` (not in panel) |
| — | `closing_radius` (not in panel) |
| — | `cell_min_area` (not in panel) |
| — | `cell_max_eccentricity` (not in panel) |
| — | `max_area` (not in panel) |
| — | `intensity_threshold` (not in panel) |
| — | `iou_threshold` (not in panel) |
| — | `max_centroid_dist` (not in panel) |
| — | `gap_frames` (not in panel) |
| — | `min_track_length` (not in panel) |

**Root cause:** `image_detection.yaml` was written for an earlier Frangi-based pipeline (`src/biohack/image_detection.py`, which is now a stub). The current pipeline (`test/filament_detection.py`) uses a Hessian-based approach with a different parameter set.

### Other Gaps

| Gap | Impact | Priority |
|---|---|---|
| Job state is in-memory only | Lost on server restart | Medium |
| No authentication on API endpoints | Anyone can upload/trigger processing | Low (internal tool) |
| Missing deps in `pyproject.toml` | `scipy`, `pandas`, `tifffile`, `flask` not declared | Low |
| nginx config not version-controlled | Manual server setup required | Low |
| No HTTPS | Traffic unencrypted | Low (internal tool) |
| Upload file size only limited by nginx (500MB) | No server-side validation of file size | Low |

## 8. Data Flow Summary

```
                    Upload
                      │
  .tif file ──POST──► Flask API ──save──► /home/ubuntu/uploads/
                      │
                      │ (background thread)
                      ▼
              filament_detection.run_pipeline()
                      │
                      │ outputs:
                      │  • *_filament_measurements.csv
                      │  • *_track_summary.csv
                      │  • *_annotated.tif
                      ▼
              export_for_frontend.export_dataset()
                      │
                      │ outputs:
                      │  • raw/frame_NNN.png (100 frames)
                      │  • annotated/frame_NNN.png (100 frames)
                      │  • structure_measurements.csv
                      │  • track_summary.csv
                      ▼
              /home/ubuntu/frontend/data/dataset_NN/
                      │
                      │ served by nginx as static files
                      ▼
              Browser fetches /data/dataset_NN/...
                      │
                      │ PapaParse CSVs, preload PNGs
                      ▼
              React renders playback + charts + tracks
```

## 9. Development & Deployment

### Local Development

```bash
# Frontend dev server
cd test/frontend
npm install
npx vite          # http://localhost:5173

# Pipeline (test locally)
cd test
python filament_detection.py ../data/ -o output/
python export_for_frontend.py   # exports to frontend/public/data/
```

### Build & Deploy

```bash
# Build frontend
cd test/frontend
npx vite build

# Deploy to EC2
rsync -avz --delete dist/ ubuntu@35.179.135.58:/home/ubuntu/frontend/ \
  -e "ssh -i ~/.ssh/HomeMBP.pem"

# Deploy pipeline code
rsync -avz test/{filament_detection.py,export_for_frontend.py,upload_server.py} \
  ubuntu@35.179.135.58:/home/ubuntu/pipeline/ \
  -e "ssh -i ~/.ssh/HomeMBP.pem"

# Restart API after pipeline changes
ssh -i ~/.ssh/HomeMBP.pem ubuntu@35.179.135.58 \
  "sudo systemctl restart upload-api"
```

### Git

- Branch: `kparmesar/frontend`
- Pipeline + export code are in `test/` (not `src/biohack/` — that's the original stub)
