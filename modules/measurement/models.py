"""
Pydantic models for ArUco-based measurement requests and responses.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class MeasurementBox(BaseModel):
    """Bounding box with segmentation mask and physical measurements."""

    label: str = Field(..., description="Class label of the object")
    confidence: float = Field(..., description="Detection confidence score")
    x1: int = Field(..., description="Left x-coordinate of bounding box")
    y1: int = Field(..., description="Top y-coordinate of bounding box")
    x2: int = Field(..., description="Right x-coordinate of bounding box")
    y2: int = Field(..., description="Bottom y-coordinate of bounding box")
    mask_base64: Optional[str] = Field(None, description="Base64-encoded mask image")
    mask_score: Optional[float] = Field(None, description="Segmentation quality score")

    # Measurements
    area_cm2: Optional[float] = Field(None, description="Area in square centimeters")
    perimeter_cm: Optional[float] = Field(None, description="Perimeter in centimeters")
    width_cm: Optional[float] = Field(None, description="Width in centimeters")
    height_cm: Optional[float] = Field(None, description="Height in centimeters")
    angle: Optional[float] = Field(None, description="Rotation angle in degrees")
    px_per_cm: Optional[float] = Field(None, description="Scale factor (pixels per cm)")


class MeasurementRequest(BaseModel):
    """Request model for object measurement."""

    image_name: str = Field(..., description="Name of the image file to process")
    boxes: List[dict] = Field(..., description="List of boxes with segmentation masks")
    marker_size_cm: float = Field(4.9, description="ArUco marker size in centimeters")
    marker_id: Optional[int] = Field(None, description="Specific marker ID to use for scale")


class MeasurementResponse(BaseModel):
    """Response model for object measurement."""

    measurements: List[MeasurementBox] = Field(..., description="List of boxes with measurements")
    markers_detected: int = Field(..., description="Number of ArUco markers detected")
    scale_px_per_cm: Optional[float] = Field(None, description="Calculated scale factor")
