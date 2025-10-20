"""
Pipeline service for orchestrating the complete image analysis workflow.
"""

from typing import List, Optional

import numpy as np
from core import DeviceManager, ImageManager
from modules.detection import DetectionBox, YOLODetector
from modules.measurement import ArucoMeasurer, MeasurementBox
from modules.segmentation import SAM2Segmentor, SegmentationBox


class PipelineService:
    """
    Orchestrates the complete image analysis pipeline:
    1. Object detection (YOLO)
    2. Segmentation (SAM2)
    3. Measurement (ArUco)
    """

    def __init__(self):
        """Initialize the pipeline with all required components."""
        # Initialize shared components
        self.device_manager = DeviceManager()
        self.image_manager = ImageManager()

        # Initialize module components
        self.yolo_detector = YOLODetector(self.device_manager)
        self.sam2_segmentor = SAM2Segmentor(self.device_manager)
        self.aruco_measurer = ArucoMeasurer()

        print("=" * 60)
        print("Pipeline Service Initialized")
        print("=" * 60)

    def detect_objects(self, model_name: str, image_name: str) -> List[DetectionBox]:
        """
        Step 1: Detect objects in an image using YOLO.

        Args:
            model_name: Name of the YOLO model to use
            image_name: Name of the image file

        Returns:
            List of DetectionBox objects
        """
        print("\n" + "=" * 60)
        print("STEP 1: Object Detection")
        print("=" * 60)

        # Load image
        image = self.image_manager.load_image(image_name)

        # Run detection
        detections = self.yolo_detector.detect_objects(image, model_name)

        print("=" * 60)
        return detections

    def segment_objects(self, model_name: str, image_name: str, boxes: List[dict]) -> List[SegmentationBox]:
        """
        Step 2: Generate segmentation masks for detected objects using SAM2.

        Args:
            model_name: Name of the SAM2 model to use
            image_name: Name of the image file
            boxes: List of bounding boxes from detection step

        Returns:
            List of SegmentationBox objects with masks
        """
        print("\n" + "=" * 60)
        print("STEP 2: Segmentation")
        print("=" * 60)

        # Load image (uses cache if same as previous step)
        image = self.image_manager.load_image(image_name)

        # Run segmentation
        segmented_boxes = self.sam2_segmentor.segment_boxes(image, boxes, model_name)

        print("=" * 60)
        return segmented_boxes

    def measure_objects(
        self,
        image_name: str,
        boxes: List[dict],
        marker_size_cm: float = 4.9,
        marker_id: Optional[int] = None,
    ) -> tuple[List[MeasurementBox], int, Optional[float]]:
        """
        Step 3: Measure objects using ArUco markers for scale.

        Args:
            image_name: Name of the image file
            boxes: List of boxes with segmentation masks
            marker_size_cm: Real-world size of ArUco marker in cm
            marker_id: Specific marker ID to use for scale

        Returns:
            Tuple of (measured_boxes, markers_detected, scale_px_per_cm)
        """
        print("\n" + "=" * 60)
        print("STEP 3: Measurement")
        print("=" * 60)

        # Load image (uses cache if same as previous step)
        image = self.image_manager.load_image(image_name)

        # Run measurement
        measured_boxes, markers_count, scale = self.aruco_measurer.measure_objects(
            image=image,
            boxes=boxes,
            marker_size_cm=marker_size_cm,
            marker_id=marker_id,
        )

        print("=" * 60)
        return measured_boxes, markers_count, scale

    def get_current_image(self) -> Optional[np.ndarray]:
        """Get the currently cached image."""
        return self.image_manager.get_current_image()

    def clear_cache(self):
        """Clear all caches (image and models)."""
        print("\n⟳ Clearing all caches...")
        self.image_manager.clear_cache()
        self.yolo_detector.clear_model()
        self.sam2_segmentor.clear_model()
        self.device_manager.clear_cache()
        print("✓ All caches cleared")

    def get_status(self) -> dict:
        """Get current status of the pipeline."""
        return {
            "device": str(self.device_manager.device),
            "current_image": self.image_manager.get_current_image_name(),
            "yolo_model": self.yolo_detector.get_current_model_name(),
            "sam2_model": self.sam2_segmentor.get_current_model_name(),
        }
