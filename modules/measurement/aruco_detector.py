"""
ArUco marker detector for real-world scale calculation.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class ArucoDetector:
    """
    Detects ArUco markers and computes real-world scale from marker size.
    """

    def __init__(self, dictionary_type: int = cv2.aruco.DICT_4X4_50):
        """
        Initialize ArUco detector.

        Args:
            dictionary_type: ArUco dictionary to use (default: DICT_4X4_50)
        """
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        self.image: Optional[np.ndarray] = None
        self.corners: Optional[List] = None
        self.marker_ids: Optional[np.ndarray] = None
        self.rejected: Optional[List] = None

    def detect_markers_in_image(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect ArUco markers in an image.

        Args:
            image: Image to process (BGR or RGB format)

        Returns:
            Tuple of (corners, marker_ids)
        """
        self.image = image

        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        self.corners, self.marker_ids, self.rejected = self.detector.detectMarkers(gray)

        return self.corners, self.marker_ids

    def get_detected_marker_info(self) -> List[Dict]:
        """
        Get information about all detected markers.

        Returns:
            List of dicts with marker_id, corners, and center coordinates
        """
        if self.marker_ids is None or len(self.marker_ids) == 0:
            return []

        markers_info = []
        for i, marker_id in enumerate(self.marker_ids.flatten()):
            corner_points = self.corners[i][0]
            center = corner_points.mean(axis=0)

            markers_info.append(
                {
                    "id": int(marker_id),
                    "corners": corner_points.tolist(),
                    "center": center.tolist(),
                }
            )

        return markers_info

    def calculate_pixels_per_cm(self, marker_size_cm: float, marker_id: Optional[int] = None) -> float:
        """
        Calculate pixels-per-centimeter scale from a detected marker.

        Args:
            marker_size_cm: Real-world size of the ArUco marker (one side, in cm)
            marker_id: Specific marker ID to use (uses first detected if None)

        Returns:
            Scale factor in pixels/cm

        Raises:
            ValueError: If no markers are detected or specified marker is not found
        """
        if self.corners is None or len(self.corners) == 0:
            raise ValueError("No ArUco markers detected in image")

        # Find the marker to use for scale calculation
        marker_index = 0
        if marker_id is not None and self.marker_ids is not None:
            matches = np.nonzero(self.marker_ids.flatten() == marker_id)[0]
            if len(matches) == 0:
                raise ValueError(f"Marker ID {marker_id} not found in detected markers")
            marker_index = matches[0]

        # Get marker corner points
        corner_points = self.corners[marker_index][0]

        # Calculate average side length in pixels
        side_lengths_px = []
        for i in range(4):
            point1 = corner_points[i]
            point2 = corner_points[(i + 1) % 4]
            length = np.linalg.norm(point2 - point1)
            side_lengths_px.append(length)

        average_side_px = np.mean(side_lengths_px)
        pixels_per_cm = average_side_px / marker_size_cm

        return float(pixels_per_cm)

    def draw_detected_markers(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw detected markers on an image.

        Args:
            image: Image to draw on (uses self.image if None)

        Returns:
            Image with markers drawn

        Raises:
            ValueError: If no image is available
        """
        if image is not None:
            result_image = image.copy()
        elif self.image is not None:
            result_image = self.image.copy()
        else:
            raise ValueError("No image available for drawing")

        if self.corners is not None and len(self.corners) > 0:
            cv2.aruco.drawDetectedMarkers(result_image, self.corners, self.marker_ids)

        return result_image

    def get_marker_count(self) -> int:
        """Get the number of detected markers."""
        if self.marker_ids is None:
            return 0
        return len(self.marker_ids)
