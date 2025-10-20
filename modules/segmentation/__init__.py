"""
Segmentation module for SAM2-based image segmentation.
"""

from .mask_utils import base64_to_mask, mask_to_base64
from .models import SegmentationBox, SegmentationRequest, SegmentationResponse
from .segmentor import SAM2Segmentor

__all__ = [
    "SAM2Segmentor",
    "SegmentationBox",
    "SegmentationRequest",
    "SegmentationResponse",
    "mask_to_base64",
    "base64_to_mask",
]
