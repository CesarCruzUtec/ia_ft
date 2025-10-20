"""
Pydantic models for image segmentation requests and responses.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SegmentationBox(BaseModel):
    """Bounding box with associated segmentation mask."""

    label: str = Field(..., description="Class label of the object")
    confidence: float = Field(..., description="Detection confidence score")
    x1: int = Field(..., description="Left x-coordinate of bounding box")
    y1: int = Field(..., description="Top y-coordinate of bounding box")
    x2: int = Field(..., description="Right x-coordinate of bounding box")
    y2: int = Field(..., description="Bottom y-coordinate of bounding box")
    mask_base64: Optional[str] = Field(None, description="Base64-encoded mask image (RGBA PNG)")
    mask_score: Optional[float] = Field(None, description="Segmentation quality score")


class SegmentationRequest(BaseModel):
    """Request model for image segmentation."""

    model_name: str = Field(..., description="Name of the SAM2 model to use")
    image_name: str = Field(..., description="Name of the image file to process")
    boxes: List[dict] = Field(..., description="List of bounding boxes to segment")


class SegmentationResponse(BaseModel):
    """Response model for image segmentation."""

    masks: List[SegmentationBox] = Field(..., description="List of boxes with segmentation masks")
