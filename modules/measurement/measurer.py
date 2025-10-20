"""
ArUco-based object measurement using segmentation masks.
"""

from typing import List, Optional

import cv2
import numpy as np

from modules.measurement.aruco_detector import ArucoDetector
from modules.measurement.measurement_utils import convert_mask_to_binary, measure_mask_dimensions
from modules.measurement.models import MeasurementBox
from modules.segmentation.mask_utils import base64_to_mask


class ArucoMeasurer:
    """
    Measures object dimensions using ArUco markers for real-world scale.
    """

    def __init__(self, dictionary_type: int = cv2.aruco.DICT_4X4_50):
        """
        Initialize ArUco measurer.

        Args:
            dictionary_type: ArUco dictionary type to use
        """
        self.aruco_detector = ArucoDetector(dictionary_type=dictionary_type)

    def measure_objects(
        self,
        image: np.ndarray,
        boxes: List[dict],
        marker_size_cm: float = 4.9,
        marker_id: Optional[int] = None,
    ) -> tuple[List[MeasurementBox], int, Optional[float]]:
        """
        Measure objects in an image using ArUco markers for scale.

        Args:
            image: Input image (RGB or BGR format)
            boxes: List of box dictionaries with mask_base64 field
            marker_size_cm: Real-world size of ArUco marker in cm
            marker_id: Specific marker ID to use (uses first if None)

        Returns:
            Tuple of (measured_boxes, markers_detected, scale_px_per_cm)

        Raises:
            ValueError: If no ArUco markers are detected or no boxes provided
        """
        if not boxes:
            raise ValueError("At least one box with mask is required for measurement")

        # Detect ArUco markers
        print("⟳ Detecting ArUco markers...")
        _, marker_ids = self.aruco_detector.detect_markers_in_image(image)

        if marker_ids is None or len(marker_ids) == 0:
            # Do not raise here — return gracefully so the pipeline can continue
            # and the UI can show a friendly message. We return no measured
            # boxes, zero markers detected, and a scale of 0.0 (px/cm).
            print("⚠️ No ArUco markers detected in image. Skipping measurement.")
            return [], 0, 0.0

        markers_count = len(marker_ids)
        print(f"✓ Found {markers_count} ArUco marker(s): {marker_ids.flatten().tolist()}")

        # Calculate scale from marker
        print(f"⟳ Calculating scale (marker size: {marker_size_cm} cm)...")
        pixels_per_cm = self.aruco_detector.calculate_pixels_per_cm(marker_size_cm=marker_size_cm, marker_id=marker_id)
        print(f"✓ Scale: {pixels_per_cm:.2f} pixels/cm")

        # Measure each object
        measured_boxes = []
        print(f"⟳ Measuring {len(boxes)} object(s)...")

        for i, box in enumerate(boxes):
            # Extract mask from base64
            if "mask_base64" not in box:
                print(f"  Warning: Box {i + 1} has no mask, skipping measurement")
                continue

            mask = base64_to_mask(box["mask_base64"])
            binary_mask = convert_mask_to_binary(mask)

            # Measure mask dimensions
            try:
                measurements = measure_mask_dimensions(binary_mask, pixels_per_cm)

                # Create MeasurementBox with all data
                measured_box = MeasurementBox(
                    label=box.get("label", "unknown"),
                    confidence=box.get("confidence", 0.0),
                    x1=box.get("x1", 0),
                    y1=box.get("y1", 0),
                    x2=box.get("x2", 0),
                    y2=box.get("y2", 0),
                    mask_base64=box.get("mask_base64"),
                    mask_score=box.get("mask_score"),
                    area_cm2=measurements.get("area_cm2"),
                    perimeter_cm=measurements.get("perimeter_cm"),
                    width_cm=measurements.get("width_cm"),
                    height_cm=measurements.get("height_cm"),
                    angle=measurements.get("angle"),
                    px_per_cm=pixels_per_cm,
                )

                measured_boxes.append(measured_box)

                print(
                    f"  Object {i + 1}: {measurements['width_cm']:.2f} x {measurements['height_cm']:.2f} cm, "
                    f"Area: {measurements['area_cm2']:.2f} cm²"
                )

            except ValueError as e:
                print(f"  Warning: Failed to measure box {i + 1}: {e}")
                continue

        print(f"✓ Successfully measured {len(measured_boxes)} object(s)")

        return measured_boxes, markers_count, pixels_per_cm

    def get_marker_info(self) -> List[dict]:
        """Get information about detected markers."""
        return self.aruco_detector.get_detected_marker_info()

    def draw_markers(self, image: np.ndarray) -> np.ndarray:
        """Draw detected ArUco markers on image."""
        return self.aruco_detector.draw_detected_markers(image)
