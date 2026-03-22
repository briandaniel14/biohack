# Filament Detection Tool — System Architecture

## 1. Overview

A web-based microscopy analysis tool that detects and tracks filamentous structures in time-lapse fluorescence microscopy images. Users upload multi-channel TIF stacks via a browser UI, the server splits channels, runs Cellpose segmentation + Frangi-based filament detection + a statistics pipeline, and results are visualised in a React frontend with playback, per-track inspection, and metrics dashboards.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User's Browser                              │
│                                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │  Upload   │  │   Tuning     │  │         Results              │  │
│  │  Page     │  │   Page       │  │  Frame playback + controls   │  │
│  │           │  │  Hyperparams │  │  Track inspector table       │  │
│  │  Upload   │  │  Preview     │  │  Metrics dashboard (4 charts)│  │
│  │  TIF      │  │  Run pipeline│  │  Click-to-select tracks      │  │
│  └─────┬─────┘  └──────┬──────┘  └──────────────┬───────────────┘  │
│        │               │                         │                  │
│        │ POST /api/upload   POST /api/run        │ GET /data/...    │
│        │ GET  /api/status   GET  /api/jobs       │ (static files)   │
│        │ GET  /api/datasets                      │                  │
└────────┼───────────────┼─────────────────────────┼──────────────────┘
         │               │                         │
         ▼               ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EC2 Instance (18.170.61.114)                      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      nginx (port 80)                         │   │
│  │                                                              │   │
│  │  /           → serve /home/ubuntu/frontend/ (SPA fallback)   │   │
│  │  /data/      → proxy to Flask (port 8080)                    │   │
│  │  /api/*      → proxy to Flask (port 8080)                    │   │
│  │  /assets/*   → 1-day cache, immutable                        │   │
│  │  index.html  → no-cache (cache-busting)                      │   │
│  │  client_max_body_size = 500M                                 │   │
│  └─────────────────────────┬────────────────────────────────────┘   │
│                            │                                        │
│  ┌─────────────────────────▼────────────────────────────────────┐   │
│  │         Flask API — upload_server.py (port 8080)             │   │
│  │         systemd service: upload-api                          │   │
│  │         Python venv: /home/ubuntu/venv                       │   │
│  │                                                              │   │
│  │  POST /api/upload      → save TIF → split channels → dataset│   │
│  │  POST /api/run         → run detection + stats pipeline      │   │
│  │  GET  /api/jobs        → list active jobs (cross-device)     │   │
│  │  GET  /api/status/:id  → poll job progress                   │   │
│  │  GET  /api/datasets    → return dataset list                 │   │
│  │  GET  /data/*          → serve dataset files (images, CSVs)  │   │
│  └─────────────────────────┬────────────────────────────────────┘   │
│                            │                                        │
│                   (background thread)                               │
│                            │                                        │
│  ┌─────────────────────────▼────────────────────────────────────┐   │
│  │  src/biohack pipeline (imported directly, not subprocess)    │   │
│  │    split_tiffs → image_detection → filament_statistics       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Infrastructure

### EC2 Instance

| Property | Value |
|---|---|
| Host | `18.170.61.114` |
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
│   └── logo.png
├── server/
│   └── upload_server.py       # Flask API server
├── src/
│   └── biohack/               # Python pipeline package
│       ├── constants.py
│       ├── experiment_config.py
│       ├── image_detection.py
│       ├── filament_statistics.py
│       ├── statistics_helper.py
│       ├── split_tiffs.py
│       └── utils.py
├── server_data/               # Dataset storage (served as /data/)
│   ├── datasets.json          # Dataset index
│   └── <dataset_id>/
│       ├── raw/               # frame_000.png … (brightfield frames)
│       ├── masks/             # frame_000.png … (Cellpose masks)
│       ├── diagnostics/       # frame_000.png … (annotated overlays)
│       ├── cell_tracks.csv    # Per-frame per-cell tracking data
│       ├── summary.json       # Run metadata (params, completed_at, run_name)
│       └── *.csv              # Additional pipeline output CSVs
├── server_uploads/            # Temporary upload staging
├── server_originals/          # Original uploaded TIF files (kept for re-runs)
└── venv/                      # Python virtual environment
```

### nginx Configuration

Version-controlled at `server/nginx-frontend.conf`:

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

    location ~ ^/data/(.+\.zip)$ {
        alias /home/ubuntu/server_data/$1;
        default_type application/octet-stream;
        add_header Content-Disposition "attachment";
        add_header Cache-Control "no-cache";
    }

    location /data/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 60s;
    }

    location /assets/ {
        expires 1d;
        add_header Cache-Control "public, immutable";
    }

    location = /index.html {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header Pragma "no-cache";
    }

    location / {
        try_files $uri $uri/ /index.html;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
}
```

Config deployed to: `/etc/nginx/sites-enabled/frontend`

### systemd Service

Version-controlled at `server/upload-api.service`:

```ini
[Unit]
Description=Filament Upload API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/venv/bin/python3 /home/ubuntu/server/upload_server.py
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

**File:** `server/upload_server.py` (deployed to `/home/ubuntu/server/upload_server.py`)
**Port:** 8080 (bound to `0.0.0.0`)

### Endpoints

#### `POST /api/upload`

Upload a multi-channel TIF stack. The server splits it into per-frame PNGs and creates a dataset.

**Request:** `multipart/form-data` with field `file`

**Validation:**
- File must be present and have a filename
- Extension must be `.tif` or `.tiff`
- Duplicate name check: returns 409 if a dataset with the same display name already exists

**Response (202):**
```json
{ "job_id": "a1b2c3d4" }
```

**Side effects:**
- Saves file to `server_uploads/`
- Starts background thread that splits channels with `split_tiffs.split_two_channel_time_stacks()`
- Creates dataset directory under `server_data/` with `raw/` frames
- Copies original TIF to `server_originals/` for re-runs
- Appends new entry to `datasets.json`

#### `POST /api/run`

Run the detection + statistics pipeline on an existing dataset.

**Request:**
```json
{
  "dataset_id": "ch20_ura7_ura8_001_crop_1",
  "params": {
    "gaussian_sigma": 1.5,
    "frangi_sigmas": [1, 2, 3, 4, 5],
    "run_name": "my experiment"
  }
}
```

**Response (202):**
```json
{ "job_id": "b5c6d7e8" }
```

**Side effects:**
- Starts background thread running `_run_pipeline_job()`
- Calls `image_detection.process_time_series_image()` for Frangi-based detection
- Calls `filament_statistics.run_filament_pipeline()` for tracking + statistics
- Generates diagnostic overlay images, CSVs, and `summary.json`

#### `GET /api/jobs`

Returns all currently-running jobs. Enables cross-device job discovery.

**Response (200):**
```json
{
  "job_id_1": { "status": "processing", "step": "Detecting filaments...", "dataset_id": "..." },
  "job_id_2": { "status": "processing", "step": "Computing statistics...", "dataset_id": "..." }
}
```

#### `GET /api/status/<job_id>`

Poll a specific job's status.

**Response (200):**
```json
{
  "status": "processing" | "complete" | "error",
  "step": "Running detection pipeline...",
  "dataset_id": "ch20_ura7_ura8_001_crop_1",
  "error": "Pipeline failed: ..."
}
```

**Note:** Job state is held **in-memory** (Python dict with a threading lock). All job state is lost on server restart.

#### `GET /api/datasets`

Returns the dataset list with result status.

**Response (200):**
```json
[
  {
    "id": "ch20_ura7_ura8_001_crop_1",
    "name": "Ch20 Ura7 Ura8 001 Crop 1",
    "frames": 50,
    "timeSpan": "12h 15m",
    "has_results": true,
    "run_name": "experiment 1",
    "completed_at": "2026-03-22T01:14:36.270401+00:00"
  }
]
```

#### `DELETE /api/dataset/<dataset_id>`

Delete a dataset and all its files.

#### `PATCH /api/dataset/<dataset_id>/run-name`

Update the run name for a dataset.

#### `DELETE /api/dataset/<dataset_id>/results`

Delete pipeline results while keeping the raw frames.

#### `POST /api/dataset/<dataset_id>/download`

Build and stream a ZIP archive of the dataset.

**Request:**
```json
{ "mode": "results" | "all" }
```

- `results`: summary.json + CSV outputs
- `all`: also includes raw/, masks/, diagnostics/ image folders

#### `POST /api/checkout`

Create a Stripe Checkout session. Stripe keys are read from environment variables (`STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`).

#### `GET /data/<path>`

Serve static dataset files (images, CSVs) from `server_data/`.

### Background Pipeline Thread

When `/api/run` is called, a daemon thread runs `_run_pipeline_job()`:

```
1. Build ExperimentConfig from params via experiment_config_from_mapping()
2. Run image_detection.process_time_series_image(raw_dir, config)
   → Frangi-based filament detection per frame
   → Produces per-frame masks and measurements
3. Run filament_statistics.run_filament_pipeline(results_dir, config)
   → Cell tracking, filament episode detection, lineage analysis
   → Produces tracking CSVs + summary statistics
4. Generate diagnostic overlay PNGs (diagnostics/ folder)
5. Write summary.json with params, run_name, completed_at, frame count
6. Update datasets.json
7. Update job status → "complete"
```

On failure: job status → `"error"` with the exception message.

## 4. Detection Pipeline

### Source Code

The pipeline lives in `src/biohack/` and is imported directly by the server (not run as a subprocess):

| Module | Purpose |
|---|---|
| `experiment_config.py` | `ExperimentConfig` dataclass + `experiment_config_from_mapping()` factory |
| `image_detection.py` | `process_time_series_image()` — Frangi-based per-frame filament detection |
| `filament_statistics.py` | `run_filament_pipeline()` — tracking, episodes, lineage, statistics |
| `statistics_helper.py` | Helper functions: `build_filament_episodes()`, `filter_pillar_tracks_v3_aggressive()` |
| `split_tiffs.py` | `split_two_channel_time_stacks()` — split multi-channel TIF into per-frame PNGs |
| `constants.py` | Structured pipeline constants, frame-filename regex |
| `utils.py` | YAML loading, image utilities |

### ExperimentConfig (key parameters)

| Category | Parameters |
|---|---|
| **Preprocessing** | `clip_low_percentile`, `clip_high_percentile`, `gaussian_sigma` |
| **Frangi detection** | `frangi_sigmas`, `frangi_threshold_percentile`, `foreground_percentile` |
| **Segmentation** | `local_block_size`, `local_offset`, `min_object_size`, `min_pixels_for_presence` |
| **Cellpose** | `model_type_bf`, `model_type_gfp`, `diameter_bf`, `diameter_gfp`, `use_gpu` |
| **Cell tracking** | `max_link_distance`, `min_overlap_fraction`, `bud_distance`, `bud_overlap_fraction` |
| **Filament tracking** | `max_filament_gap` |
| **Pillar filtering** | `remove_pillar_tracks`, `pillar_v3_*` parameters |
| **Output** | `run_name`, `dataset_directory`, `results_directory`, `verbose`, `figure_dpi`, `cmap` |

### Pipeline Stages

```
Multi-channel TIF Stack
  │
  ▼
┌───────────────────────┐
│ 1. Channel Splitting   │  split_tiffs → brightfield + GFP per-frame PNGs
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 2. Cell Segmentation   │  Cellpose on brightfield → cell masks
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 3. Frangi Detection    │  Multi-scale Frangi filter on GFP channel
│                        │  → filament response map → threshold → label
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 4. Cell Tracking       │  IoU + centroid linking across frames
│                        │  Bud detection + mother-daughter linking
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 5. Filament Episodes   │  Map filaments to cells, build episodes
│                        │  Track appearance/disappearance per cell
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 6. Statistics          │  Per-cell, per-filament-episode summaries
│    & Lineage           │  Cell lineage table, pillar track filtering
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 7. Diagnostic Output   │  Overlay PNGs, tracking CSVs, summary.json
└───────────────────────┘
```

### Pipeline Outputs (per dataset)

| File/Directory | Description |
|---|---|
| `raw/frame_NNN.png` | Brightfield frames (grayscale PNG) |
| `masks/frame_NNN.png` | Cellpose segmentation masks |
| `diagnostics/frame_NNN.png` | Annotated overlays with detections |
| `cell_tracks.csv` | Per-frame per-cell tracking data with filament columns |
| `filament_episodes.csv` | Filament appearance episodes per cell |
| `filament_episodes_with_lineage.csv` | Episodes with mother cell lineage |
| `filaments_per_frame.csv` | Frame-level filament counts |
| `summary.json` | Run metadata: params, timestamps, frame count |

## 5. Frontend

**Stack:** React 19, Vite 6, Tailwind CSS 4, Recharts, PapaParse
**Source:** `frontend/`
**Build:** `cd frontend && npx vite build` → `frontend/dist/`

### Three-Page Architecture

```
App.jsx (shell)
├── Header: logo + title + nav tabs [Upload | Tuning | Results]
├── Shared state: datasets[], currentDataset, pipelineJobs, page
│
├── UploadPage.jsx
│   ├── Upload zone (drag & drop .tif/.tiff)
│   ├── Dataset list with delete (double-click confirm)
│   └── Pipeline run results list with delete
│
├── TuningPage.jsx
│   ├── Left (42%): frame preview with scrubber (raw/mask/diagnostic views)
│   └── Right (58%): HyperparamPanel + "Run Pipeline" button
│       └── Run button disabled while pipeline is running
│
└── ResultsPage.jsx
    ├── Left (42%): frame player with full playback controls
    └── Right (58%):
        ├── TrackInspector.jsx: sortable track table + detail panel
        └── MetricsDashboard.jsx: 2×2 chart grid
```

### Source Files

| File | Purpose |
|---|---|
| `App.jsx` | Shell — shared state, header with nav tabs, pipeline job management, cross-device job discovery |
| `UploadPage.jsx` | Upload zone, dataset list, run list, double-click delete confirmation |
| `TuningPage.jsx` | Frame preview with view modes, hyperparameter panel, pipeline trigger |
| `ResultsPage.jsx` | Playback, track selection, inspector, metrics |
| `TrackInspector.jsx` | Sortable track table, selected track detail with sparklines |
| `MetricsDashboard.jsx` | 2×2 chart grid (length, count, eccentricity, axes), 1-indexed frames |
| `HyperparamPanel.jsx` | Parameter sliders/inputs with tooltips, exposes values via ref |
| `PricingModal.jsx` | Stripe pricing modal with checkout flow |
| `data.js` | API helpers, data loading, frame URLs, track colours, constants |
| `main.jsx` | React entry point |
| `index.css` | Tailwind directives |

### Job Management

Pipeline jobs support concurrent runs and cross-device visibility:

- **Client-side:** `pipelineJobs` state persisted to `sessionStorage` (survives tab refresh)
- **Server-side:** In-memory `_jobs` dict with thread-safe lock
- **Cross-device discovery:** On mount, frontend calls `GET /api/jobs` to discover running jobs from other devices/tabs
- **Polling:** 2-second intervals per active job via `GET /api/status/<job_id>`
- **Duplicate prevention:** Only one pipeline run per dataset at a time

### Data Loading

Dataset viewing uses static file fetches served by Flask:

- `GET /data/<dataset_id>/summary.json` → run metadata
- `GET /data/<dataset_id>/cell_tracks.csv` → parsed with PapaParse
- `GET /data/<dataset_id>/{raw|masks|diagnostics}/frame_NNN.png` → frame images

Frame images are preloaded on dataset switch for all three view modes.

### Upload Flow

```
User drags .tif onto upload zone
         │
         ▼
UploadPage validates extension (.tif/.tiff)
         │
         ▼
POST /api/upload (FormData with file)
         │
         ├── 409: duplicate name → show error
         │
         ▼
Server saves, splits channels, returns { job_id }
         │
         ▼
UploadPage polls GET /api/status/{job_id} every 2 seconds
         │
         ├── status: "processing"  → show step text in UI
         ├── status: "error"       → show error, stop polling
         └── status: "complete"    → refresh dataset list,
                                     select new dataset
```

### Pipeline Run Flow

```
User adjusts hyperparams on Tuning page → clicks "Run Pipeline"
         │
         ▼
POST /api/run { dataset_id, params }
         │
         ▼
Server returns { job_id }, starts background thread
         │
         ▼
App.jsx adds to pipelineJobs, starts 2s polling
         │
         ├── On other devices: GET /api/jobs discovers the running job
         │
         ├── status: "processing"  → show step text, disable Run button
         ├── status: "error"       → show error, auto-clear after 5s
         └── status: "complete"    → refresh datasets, auto-clear after 5s
```

## 6. Data Flow Summary

```
                    Upload
                      │
  .tif file ──POST──► Flask API ──save──► server_uploads/
                      │
                      │ (background thread)
                      ▼
              split_tiffs.split_two_channel_time_stacks()
                      │
                      │ outputs:
                      │  • raw/frame_NNN.png (brightfield)
                      │  • GFP frames for detection
                      ▼
              server_data/<dataset_id>/
                      │
                    Run Pipeline
                      │
  params ────POST───► Flask API ──► background thread
                      │
                      ▼
              image_detection.process_time_series_image()
                      │
                      │ outputs: masks, filament measurements
                      ▼
              filament_statistics.run_filament_pipeline()
                      │
                      │ outputs:
                      │  • masks/frame_NNN.png
                      │  • diagnostics/frame_NNN.png
                      │  • cell_tracks.csv
                      │  • filament_episodes*.csv
                      │  • summary.json
                      ▼
              server_data/<dataset_id>/
                      │
                      │ served by Flask as /data/<dataset_id>/...
                      ▼
              Browser fetches /data/<dataset_id>/...
                      │
                      │ PapaParse CSVs, preload PNGs
                      ▼
              React renders playback + charts + tracks
```

## 7. Development & Deployment

### Local Development

```bash
# Frontend dev server
cd frontend
npm install
npx vite          # http://localhost:5173

# Run pipeline locally
.venv/bin/python scripts/example.py
```

### Build & Deploy

```bash
# Build frontend
cd frontend
npx vite build

# Deploy frontend to EC2
rsync -avz --delete --exclude='data/' frontend/dist/ \
  ubuntu@18.170.61.114:/home/ubuntu/frontend/ \
  -e "ssh -i ~/.ssh/HomeMBP.pem"

# Deploy server
scp -i ~/.ssh/HomeMBP.pem server/upload_server.py \
  ubuntu@18.170.61.114:/home/ubuntu/server/upload_server.py

# Deploy pipeline code
rsync -avz -e "ssh -i ~/.ssh/HomeMBP.pem" \
  src/biohack/ ubuntu@18.170.61.114:/home/ubuntu/src/biohack/

# Restart API after changes
ssh -i ~/.ssh/HomeMBP.pem ubuntu@18.170.61.114 \
  "sudo systemctl restart upload-api"
```

### Git

- Branch: `frontend` — UI + server + pipeline changes
- Branch: `fear/dbscan-detection` — base pipeline work
- Pipeline code: `src/biohack/`
- Server code: `server/`
- Frontend code: `frontend/`

## 8. Known Gaps

| Gap | Impact | Priority |
|---|---|---|
| Job state is in-memory only | Lost on server restart | Medium |
| No authentication on API endpoints | Anyone can upload/trigger processing | Low (internal tool) |
| No HTTPS | Traffic unencrypted | Low (internal tool) |
| Stripe keys via env vars only | Must be set on server | Low |
