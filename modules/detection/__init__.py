"""
Detection module for YOLO-based object detection.
"""

from .detector import YOLODetector
from .models import DetectionBox, DetectionRequest, DetectionResponse

__all__ = [
    "YOLODetector",
    "DetectionBox",
    "DetectionRequest",
    "DetectionResponse",
]
