"""
Pydantic models for object detection requests and responses.
"""

from typing import List

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    """Represents a single detected object with bounding box."""

    label: str = Field(..., description="Class label of the detected object")
    confidence: float = Field(..., description="Detection confidence score (0-1)")
    x1: int = Field(..., description="Left x-coordinate of bounding box")
    y1: int = Field(..., description="Top y-coordinate of bounding box")
    x2: int = Field(..., description="Right x-coordinate of bounding box")
    y2: int = Field(..., description="Bottom y-coordinate of bounding box")


class DetectionRequest(BaseModel):
    """Request model for object detection."""

    model_name: str = Field(..., description="Name of the YOLO model to use")
    image_name: str = Field(..., description="Name of the image file to process")


class DetectionResponse(BaseModel):
    """Response model for object detection."""

    detections: List[DetectionBox] = Field(..., description="List of detected objects")
