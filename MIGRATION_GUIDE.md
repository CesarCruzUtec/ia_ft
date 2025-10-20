# Migration Guide: Refactored Backend Structure

## Overview
The backend has been refactored from a monolithic structure to a modular, object-oriented architecture following best practices.

## What Changed

### Old Structure (Before)
```
back/
├── main.py              # FastAPI routes + all logic
├── utils.py             # ImageAnalyzer class (267 lines, too many responsibilities)
└── aruco/               # Standalone ArUco module
    ├── cli.py
    ├── main.py
    └── src/
        ├── detector.py
        └── utils.py
```

### New Structure (After)
```
ia_ft/                               # Project root (backend moved here)
├── main.py                          # FastAPI routes only (clean)
├── config.py                        # Configuration & constants
│
├── core/                            # Shared components
│   ├── device_manager.py            # GPU/CPU device management
│   └── image_manager.py             # Image loading & caching
│
├── modules/                         # Feature modules
│   ├── detection/                   # YOLO object detection
│   │   ├── detector.py              # YOLODetector class
│   │   └── models.py                # Pydantic models
│   │
│   ├── segmentation/                # SAM2 segmentation
│   │   ├── segmentor.py             # SAM2Segmentor class
│   │   ├── mask_utils.py            # Mask conversion utilities
│   │   └── models.py                # Pydantic models
│   │
│   └── measurement/                 # ArUco measurement
│       ├── measurer.py              # ArucoMeasurer class
│       ├── aruco_detector.py        # ArUco detection logic
│       ├── measurement_utils.py     # Measurement utilities
│       └── models.py                # Pydantic models
│
├── services/                        # Business logic
│   └── pipeline_service.py          # Orchestrates all modules
│
├── utils/                           # General utilities
│   └── helpers.py                   # Helper functions
│
├── webui/                           # Gradio web UI for local testing
│   ├── app.py                       # Gradio interface
│   └── requirements.txt             # Gradio dependencies
│
├── images/                          # Image data directory
├── models/                          # YOLO model files
└── meta-sam2/                       # SAM2 checkpoints & configs
```

## API Endpoint Changes

### Renamed Endpoints
| Old Endpoint     | New Endpoint | Description                    |
|------------------|--------------|--------------------------------|
| `/get_boxes`     | `/detect`    | Object detection with YOLO     |
| `/analyze`       | `/segment`   | Segmentation with SAM2         |
| `/measure`       | `/measure`   | Measurement with ArUco (same)  |

### Updated Request/Response Models

#### Detection (formerly `get_boxes`)
**Request:**
```python
{
    "model_name": "detection_model",  # renamed from "model"
    "image_name": "image.jpg"
}
```

**Response:**
```python
{
    "detections": [...]  # same structure
}
```

#### Segmentation (formerly `analyze`)
**Request:**
```python
{
    "model_name": "sam2.1_hiera_tiny",  # renamed from "model"
    "image_name": "image.jpg",
    "boxes": [...]
}
```

**Response:**
```python
{
    "masks": [...]  # renamed from top-level structure
}
```

#### Measurement
**Request:** (unchanged structure, added optional fields)
```python
{
    "image_name": "image.jpg",
    "boxes": [...],
    "marker_size_cm": 4.9,      # optional, default 4.9
    "marker_id": null            # optional, uses first if not specified
}
```

**Response:**
```python
{
    "measurements": [...],
    "markers_detected": 2,
    "scale_px_per_cm": 45.2
}
```

## Code Improvements

### 1. Better Naming Conventions
- `get_boxes()` → `detect_objects()`
- `analyze_image()` → `segment_objects()`
- `load_model()` → Specific: `load_model()` for SAM2, YOLO
- `mask_to_base64()` → More descriptive location
- Variables use full words: `px_per_cm` instead of abbreviations

### 2. Separation of Concerns
- **DeviceManager**: Handles CUDA/MPS/CPU selection (was scattered in ImageAnalyzer)
- **ImageManager**: Handles image loading/caching (extracted from ImageAnalyzer)
- **YOLODetector**: Only detection logic
- **SAM2Segmentor**: Only segmentation logic
- **ArucoMeasurer**: Only measurement logic
- **PipelineService**: Orchestrates all modules

### 3. Single Responsibility Principle
Each class now has ONE clear responsibility:
- ✓ `DeviceManager` → Device selection
- ✓ `ImageManager` → Image management
- ✓ `YOLODetector` → Object detection
- ✓ `SAM2Segmentor` → Segmentation
- ✓ `ArucoMeasurer` → Measurement
- ✓ `PipelineService` → Orchestration

### 4. Dependency Injection
Components share resources via injection:
```python
device_manager = DeviceManager()  # Singleton
yolo_detector = YOLODetector(device_manager)  # Inject shared device
sam2_segmentor = SAM2Segmentor(device_manager)  # Inject shared device
```

### 5. Better Error Handling
- Uses proper HTTP status codes (404, 400, 500)
- More descriptive error messages
- Validation at API layer

### 6. Type Safety
- All models use Pydantic for validation
- Type hints throughout the codebase
- Better IDE autocomplete support

## Migration Steps for Frontend

### 1. Update Endpoint URLs
```javascript
// Old
POST /get_boxes
POST /analyze
POST /measure

// New
POST /detect
POST /segment
POST /measure
```

### 2. Update Request Field Names
```javascript
// Old
{ model: "...", image_name: "..." }

// New
{ model_name: "...", image_name: "..." }
```

### 3. Update Response Parsing
```javascript
// Segmentation endpoint
// Old: response.masks
// New: response.masks (same)

// Detection endpoint
// Old: response.detections
// New: response.detections (same)
```

## Benefits of Refactoring

1. **Maintainability**: Each module can be updated independently
2. **Testability**: Each component can be unit tested in isolation
3. **Readability**: Clear structure and naming conventions
4. **Scalability**: Easy to add new features (e.g., classification module)
5. **Reusability**: Modules can be used in other projects
6. **Performance**: Shared device manager prevents redundant GPU operations
7. **Documentation**: Better code organization aids understanding

## Backward Compatibility Notes

### Breaking Changes
1. API endpoint names changed (`/get_boxes` → `/detect`, `/analyze` → `/segment`)
2. Request field `model` renamed to `model_name`
3. Response structure slightly changed for measurement endpoint

### Non-Breaking Changes
- Internal refactoring doesn't affect API behavior
- Same functionality, better organization
- Performance improvements from shared resources

## Next Steps

1. Update frontend to use new endpoint names
2. Test all three endpoints with existing images
3. Consider adding new endpoints:
   - `/status` - Get pipeline status
   - `/clear-cache` - Clear model/image caches
   - `/health` - Health check endpoint

## Questions?

If you encounter any issues during migration, check:
1. Endpoint URLs are updated
2. Request field names match new schema
3. Response parsing handles new structure
