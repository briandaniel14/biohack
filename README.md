# Filament Fillies

A web-based microscopy analysis tool available at www.filament-tracker.co.uk for detecting and tracking filamentous structures in time-lapse fluorescence microscopy images. The system provides a pipeline for segmentation, classification, and visualization of filaments, with a modern React frontend and Python backend.

## Features

Upload multi-channel TIF stacks via a browser UI
Automated channel splitting, segmentation (Cellpose), and Frangi-based filament detection
Filament classification (length, lifecycle, etc.)
Interactive results visualization: playback, per-track inspection, and metrics dashboards
Hyperparameter tuning and metrics dashboard
Synthetic data generation and standard dataset support

## System Architecture

Frontend: React (located in frontend)
Backend: Python (Flask API, main code in biohack)
Data: Input/output in data, including synthetic and live datasets
Deployment: Nginx serves the frontend and proxies API/data requests to Flask

## Quickstart

```
git clone https://github.com/briandaniel14/biohack
cd biohack

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## To start the frontend (from frontend/):

```
cd frontend
npm install
npm run dev
```

## Usage

Upload TIF stacks via the web UI
Run the detection pipeline and inspect results
Tune hyperparameters and view metrics