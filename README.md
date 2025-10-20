# Image Analysis Pipeline

> Modular pipeline for object detection, segmentation, and measurement using YOLO, SAM2, and ArUco markers.
> Includes both a FastAPI backend and a Gradio WebUI for local testing.

## 🏗️ Architecture

### Overview
The backend follows a **modular, object-oriented architecture** with clear separation of concerns:

```
API Layer (main.py)
    ↓
Service Layer (pipeline_service.py)
    ↓
┌─────────────┬──────────────┬───────────────┐
│  Detection  │ Segmentation │  Measurement  │
│   (YOLO)    │    (SAM2)    │    (ArUco)    │
└─────────────┴──────────────┴───────────────┘
    ↓               ↓               ↓
Core Layer (DeviceManager, ImageManager)
```

### Directory Structure

```
ia_ft/                               # Project root
├── main.py                          # FastAPI application & routes
├── config.py                        # Configuration & constants
│
├── core/                            # Shared core components
│   ├── device_manager.py            # GPU/CPU device management
│   └── image_manager.py             # Image loading & caching
│
├── modules/                         # Feature modules
│   ├── detection/                   # YOLO object detection
│   │   ├── detector.py              # YOLODetector class
│   │   └── models.py                # Pydantic request/response models
│   │
│   ├── segmentation/                # SAM2 image segmentation
│   │   ├── segmentor.py             # SAM2Segmentor class
│   │   ├── mask_utils.py            # Mask conversion utilities
│   │   └── models.py                # Pydantic models
│   │
│   └── measurement/                 # ArUco-based measurement
│       ├── measurer.py              # ArucoMeasurer class
│       ├── aruco_detector.py        # ArUco marker detection
│       ├── measurement_utils.py     # Measurement utilities
│       └── models.py                # Pydantic models
│
├── services/                        # Business logic orchestration
│   └── pipeline_service.py          # PipelineService (orchestrates modules)
│
├── utils/                           # General utilities
│   └── helpers.py                   # Helper functions
│
├── webui/                           # Gradio web UI for local testing
│   ├── app.py                       # Gradio interface
│   ├── requirements.txt             # Gradio dependencies
│   └── README.md                    # WebUI usage guide
│
├── images/                          # Image data directory
├── models/                          # YOLO model files
└── meta-sam2/                       # SAM2 checkpoints & configs
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- GPU with CUDA support (optional, but recommended)
- `uv` package manager (recommended) or `pip`

### Installation

```bash
# Clone the repository (if not already cloned)
# Include submodules (SAM2) on first clone
git clone --recurse-submodules <your-repo-url>
cd ia_ft

# If you already cloned without submodules, initialize and update:
# git submodule update --init --recursive

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Running the API Server

Start the FastAPI backend server:

```bash
# Using uv (recommended)
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using fastapi CLI
fastapi dev main.py
```

The API will be available at:
- **API Docs:** `http://localhost:8000/docs`
- **API Root:** `http://localhost:8000`

### Running the Gradio WebUI (Local Testing)

For local testing without needing to set up the full API, you can use the Gradio interface:

```bash
# Make sure dependencies are installed
uv sync

# Ensure submodule (SAM2) is pulled and checkpoints are present
# If needed, (re)initialize submodules:
# git submodule update --init --recursive

# Run the WebUI as a module from the project root
uv run python -m webui.app

# Or without uv (after installing dependencies)
python -m webui.app
```

The WebUI will be available at `http://localhost:7860`

**WebUI Features:**
- 📤 Upload images directly from your browser
- 🎯 Select YOLO detection model
- 🎨 Select SAM2 segmentation model
- 📏 Configure ArUco marker size
- ✨ Run full pipeline with one click
- 👁️ View results and mask overlays instantly

## 📡 API Endpoints

### 1. Object Detection
**Endpoint:** `POST /detect`

Detect objects in an image using YOLO.

**Request:**
```json
{
  "model_name": "detection_model",
  "image_name": "potato.jpg"
}
```

**Response:**
```json
{
  "detections": [
    {
      "label": "potato",
      "confidence": 0.95,
      "x1": 100,
      "y1": 150,
      "x2": 300,
      "y2": 350
    }
  ]
}
```

### 2. Segmentation
**Endpoint:** `POST /segment`

Generate segmentation masks for detected objects using SAM2.

**Request:**
```json
{
  "model_name": "sam2.1_hiera_tiny",
  "image_name": "potato.jpg",
  "boxes": [
    {
      "label": "potato",
      "confidence": 0.95,
      "x1": 100,
      "y1": 150,
      "x2": 300,
      "y2": 350
    }
  ]
}
```

**Response:**
```json
{
  "masks": [
    {
      "label": "potato",
      "confidence": 0.95,
      "x1": 100,
      "y1": 150,
      "x2": 300,
      "y2": 350,
      "mask_base64": "data:image/png;base64,...",
      "mask_score": 0.98
    }
  ]
}
```

### 3. Measurement
**Endpoint:** `POST /measure`

Measure objects using ArUco markers for real-world scale.

**Request:**
```json
{
  "image_name": "potato.jpg",
  "boxes": [
    {
      "label": "potato",
      "confidence": 0.95,
      "x1": 100,
      "y1": 150,
      "x2": 300,
      "y2": 350,
      "mask_base64": "data:image/png;base64,...",
      "mask_score": 0.98
    }
  ],
  "marker_size_cm": 4.9,
  "marker_id": null
}
```

**Response:**
```json
{
  "measurements": [
    {
      "label": "potato",
      "confidence": 0.95,
      "x1": 100,
      "y1": 150,
      "x2": 300,
      "y2": 350,
      "mask_base64": "data:image/png;base64,...",
      "mask_score": 0.98,
      "area_cm2": 45.2,
      "perimeter_cm": 28.5,
      "width_cm": 8.2,
      "height_cm": 6.5,
      "angle": 12.5,
      "px_per_cm": 45.2
    }
  ],
  "markers_detected": 2,
  "scale_px_per_cm": 45.2
}
```

### Utility Endpoints

**Get Status:** `GET /status`
```json
{
  "device": "cuda",
  "current_image": "potato.jpg",
  "yolo_model": "detection_model",
  "sam2_model": "sam2.1_hiera_tiny"
}
```

**Clear Cache:** `POST /clear-cache`
```json
{
  "status": "success",
  "message": "All caches cleared"
}
```

## 🧩 Core Components

### DeviceManager
Singleton manager for GPU/CPU device selection and configuration.
- Auto-selects best available device (CUDA > MPS > CPU)
- Configures device-specific optimizations (TF32 for Ampere+ GPUs)
- Provides cache clearing utilities

### ImageManager
Manages image loading and caching.
- Searches for images recursively in the images directory
- Caches loaded images to avoid redundant I/O
- Converts images to RGB numpy arrays

### YOLODetector
YOLO-based object detector with model caching.
- Loads and caches YOLO models
- Runs inference on images
- Returns structured detection results

### SAM2Segmentor
SAM2 segmentation with model caching.
- Loads and caches SAM2 models
- Generates masks from bounding boxes
- Converts masks to base64-encoded PNG images

### ArucoMeasurer
ArUco marker-based measurement system.
- Detects ArUco markers in images
- Calculates real-world scale (pixels per cm)
- Measures object dimensions from masks

### PipelineService
Orchestrates the complete pipeline.
- Manages all module instances
- Coordinates shared resources
- Provides high-level API for the three-step workflow

## 🔧 Configuration

Edit `config.py` to customize:
- Directory paths (models, images, SAM2 checkpoints)
- SAM2 model configurations
- ArUco marker settings
- Mask visualization parameters

## 📊 Workflow

The typical workflow involves three steps:

1. **Detection** → Get bounding boxes of objects
2. **Segmentation** → Generate precise masks for each object
3. **Measurement** → Calculate real-world dimensions using ArUco markers

Each step can be called independently via the API.

## 🎯 Design Principles

1. **Single Responsibility**: Each class has one clear purpose
2. **Dependency Injection**: Shared resources injected into modules
3. **Caching**: Models and images cached to optimize performance
4. **Type Safety**: Pydantic models for validation
5. **Modularity**: Features can be added/removed independently
6. **Testability**: Components can be unit tested in isolation

## 🛠️ Development

### Project Setup

All backend code is now in the root directory for simplified development. The frontend has been removed in favor of the Gradio WebUI for local testing.

### Adding a New Module

1. Create module directory under `modules/`
2. Implement main class (e.g., `ClassifierModule`)
3. Create Pydantic models in `models.py`
4. Add module to `PipelineService`
5. Create API endpoint in `main.py`

### Running Tests

```bash
# Test all imports and initialization
uv run python test_refactoring.py

# Run unit tests (when available)
pytest tests/
```

---

For migration from the old structure, see [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)

## 📝 Notes

- Images are searched recursively in the `./images/` directory
- YOLO models should be in `./models/` directory (`.pt` files)
- SAM2 lives in `./meta-sam2` as a git submodule. If missing, run:
  - `git submodule update --init --recursive`
- SAM2 checkpoints should be in `./meta-sam2/checkpoints/`
- The server auto-selects the best available device (GPU/CPU)
- For local testing, use the Gradio WebUI in `./webui/`

## 🗂️ Project Structure

```
ia_ft/                    # Project root
├── main.py              # FastAPI application
├── config.py            # Configuration
├── core/                # Core components
├── modules/             # Feature modules
├── services/            # Business logic
├── utils/               # Utilities
├── webui/               # Gradio WebUI
├── images/              # Image data
├── models/              # YOLO models
└── meta-sam2/           # SAM2 git submodule and checkpoints
```

For migration from the old structure, see [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)
