"""
FastAPI application for image analysis pipeline.

Provides endpoints for:
1. Object detection with YOLO
2. Segmentation with SAM2
3. Measurement with ArUco markers
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from modules.detection.models import DetectionRequest, DetectionResponse
from modules.measurement.models import MeasurementRequest, MeasurementResponse
from modules.segmentation.models import SegmentationRequest, SegmentationResponse
from services import PipelineService
from utils import print_dict

# Initialize FastAPI application
app = FastAPI(
    title="Image Analysis Pipeline API",
    description="API for object detection, segmentation, and measurement using YOLO, SAM2, and ArUco",
    version="2.0.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline service (singleton)
pipeline_service = PipelineService()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Image Analysis Pipeline API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "detection": "/detect",
            "segmentation": "/segment",
            "measurement": "/measure",
            "status": "/status",
        },
    }


@app.get("/status")
async def get_status():
    """Get current status of the pipeline."""
    return pipeline_service.get_status()


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    """
    Step 1: Detect objects in an image using YOLO.

    Args:
        request: Detection request with model_name and image_name

    Returns:
        DetectionResponse with list of detected objects
    """
    print("\n" + "=" * 60)
    print("API REQUEST: /detect")
    print("=" * 60)
    print_dict(request.model_dump())

    try:
        detections = pipeline_service.detect_objects(
            model_name=request.model_name,
            image_name=request.image_name,
        )

        return DetectionResponse(detections=detections)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/segment", response_model=SegmentationResponse)
async def segment_objects(request: SegmentationRequest):
    """
    Step 2: Generate segmentation masks for detected objects using SAM2.

    Args:
        request: Segmentation request with model_name, image_name, and boxes

    Returns:
        SegmentationResponse with boxes and segmentation masks
    """
    print("\n" + "=" * 60)
    print("API REQUEST: /segment")
    print("=" * 60)
    print_dict(request.model_dump())

    try:
        if not request.boxes:
            raise ValueError("At least one bounding box is required for segmentation")

        segmented_boxes = pipeline_service.segment_objects(
            model_name=request.model_name,
            image_name=request.image_name,
            boxes=request.boxes,
        )

        return SegmentationResponse(masks=segmented_boxes)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/measure", response_model=MeasurementResponse)
async def measure_objects(request: MeasurementRequest):
    """
    Step 3: Measure objects using ArUco markers for real-world scale.

    Args:
        request: Measurement request with image_name, boxes, and marker info

    Returns:
        MeasurementResponse with physical measurements
    """
    print("\n" + "=" * 60)
    print("API REQUEST: /measure")
    print("=" * 60)
    print_dict(request.model_dump())

    try:
        if not request.boxes:
            raise ValueError("At least one box with mask is required for measurement")

        measured_boxes, markers_count, scale = pipeline_service.measure_objects(
            image_name=request.image_name,
            boxes=request.boxes,
            marker_size_cm=request.marker_size_cm,
            marker_id=request.marker_id,
        )

        return MeasurementResponse(
            measurements=measured_boxes,
            markers_detected=markers_count,
            scale_px_per_cm=scale,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Measurement failed: {str(e)}")


@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches (image and models)."""
    pipeline_service.clear_cache()
    return {"status": "success", "message": "All caches cleared"}
