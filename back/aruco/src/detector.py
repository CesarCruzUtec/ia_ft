from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class ArucoDetector:
    """Detects ArUco markers and computes real-world scale."""

    def __init__(self, dictionary_type=cv2.aruco.DICT_4X4_50):
        """
        Initialize ArUco detector.

        Args:
            dictionary_type: ArUco dictionary to use (default: DICT_4X4_50)
        """
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        self.image = None
        self.corners = None
        self.ids = None
        self.rejected = None

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path."""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.image

    def detect_markers(self, image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect ArUco markers in image.

        Args:
            image: Image to process (uses self.image if None)

        Returns:
            Tuple of (corners, ids)
        """
        if image is not None:
            self.image = image

        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.corners, self.ids, self.rejected = self.detector.detectMarkers(gray)

        return self.corners, self.ids

    def get_marker_info(self) -> List[Dict]:
        """
        Get information about detected markers.

        Returns:
            List of dicts with marker_id, corners, center
        """
        if self.ids is None:
            return []

        markers = []
        for i, marker_id in enumerate(self.ids.flatten()):
            corner = self.corners[i][0]
            center = corner.mean(axis=0)
            markers.append({"id": int(marker_id), "corners": corner.tolist(), "center": center.tolist()})
        return markers

    def calculate_scale(self, marker_size_cm: float, marker_id: Optional[int] = None) -> float:
        """
        Calculate pixels-per-cm scale from detected marker.

        Args:
            marker_size_cm: Real-world size of the ArUco marker (one side, in cm)
            marker_id: Specific marker ID to use (uses first detected if None)

        Returns:
            Scale factor in pixels/cm
        """
        if self.corners is None or len(self.corners) == 0:
            raise ValueError("No markers detected. Call detect_markers() first.")

        # Find the marker to use
        idx = 0
        if marker_id is not None and self.ids is not None:
            matches = np.nonzero(self.ids.flatten() == marker_id)[0]
            if len(matches) == 0:
                raise ValueError(f"Marker ID {marker_id} not found")
            idx = matches[0]

        # Get marker corners
        corners = self.corners[idx][0]

        # Calculate average side length in pixels
        side_lengths = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            side_lengths.append(length)

        avg_side_px = np.mean(side_lengths)
        px_per_cm = avg_side_px / marker_size_cm

        return px_per_cm

    def draw_markers(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw detected markers on image.

        Args:
            image: Image to draw on (uses self.image if None)

        Returns:
            Image with markers drawn
        """
        if image is not None:
            img = image.copy()
        elif self.image is not None:
            img = self.image.copy()
        else:
            raise ValueError("No image available")

        if self.corners is not None and len(self.corners) > 0:
            cv2.aruco.drawDetectedMarkers(img, self.corners, self.ids)

        return img
