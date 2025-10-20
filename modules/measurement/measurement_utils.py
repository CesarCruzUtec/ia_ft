"""
Utility functions for mask measurement and visualization.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def convert_mask_to_binary(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Convert any mask format to binary (0 or 255).

    Args:
        mask: Input mask (can be 2D, 3D, tensor, etc.)
        threshold: Binarization threshold

    Returns:
        Binary mask (uint8, values 0 or 255)
    """
    # Handle torch tensors
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()

    # Convert to numpy if needed
    mask = np.array(mask)

    # Handle 3D masks (take first channel)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]

    # Normalize to 0-255 range
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # Apply binary thresholding
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    return binary_mask


def measure_mask_dimensions(mask: np.ndarray, pixels_per_cm: Optional[float] = None) -> Dict:
    """
    Measure dimensions of a binary mask in pixels and optionally in centimeters.

    Args:
        mask: Binary mask (uint8, values 0 or 255)
        pixels_per_cm: Scale factor (pixels per cm). If None, only pixel measurements returned.

    Returns:
        Dictionary with measurements:
            - area_px, perimeter_px (in pixels)
            - width_px, height_px (in pixels)
            - angle (rotation angle in degrees)
            - center_x, center_y (centroid coordinates)
            - area_cm2, perimeter_cm, width_cm, height_cm (if scale provided)
            - px_per_cm (if scale provided)

    Raises:
        ValueError: If no contours are found in the mask
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No contours found in mask - mask may be empty")

    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate pixel-based measurements
    area_px = cv2.contourArea(largest_contour)
    perimeter_px = cv2.arcLength(largest_contour, closed=True)

    # Get minimum area rectangle (rotated bounding box)
    rect = cv2.minAreaRect(largest_contour)
    (center_x, center_y), (width, height), angle = rect

    # Ensure width >= height by convention
    if width < height:
        width, height = height, width
        angle = (angle + 90) % 180

    # Build result dictionary
    measurements = {
        "area_px": float(area_px),
        "perimeter_px": float(perimeter_px),
        "width_px": float(width),
        "height_px": float(height),
        "angle": float(angle),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "contour_points": len(largest_contour),
    }

    # Add centimeter measurements if scale is provided
    if pixels_per_cm is not None:
        measurements["area_cm2"] = area_px / (pixels_per_cm**2)
        measurements["perimeter_cm"] = perimeter_px / pixels_per_cm
        measurements["width_cm"] = width / pixels_per_cm
        measurements["height_cm"] = height / pixels_per_cm
        measurements["px_per_cm"] = pixels_per_cm

    return measurements


def draw_measurements_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    measurements: Dict,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw measurement annotations on an image.

    Args:
        image: Input image (BGR format)
        mask: Binary mask
        measurements: Dictionary from measure_mask_dimensions()
        color: Color for drawing annotations (BGR)

    Returns:
        Image with measurement annotations drawn
    """
    annotated_image = image.copy()

    # Draw mask contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(annotated_image, [largest_contour], -1, color, 2)

    # Draw oriented bounding box
    center_x = measurements["center_x"]
    center_y = measurements["center_y"]
    width = measurements["width_px"]
    height = measurements["height_px"]
    angle = measurements["angle"]

    box_points = cv2.boxPoints(((center_x, center_y), (width, height), angle))
    box_points = np.int0(box_points)
    cv2.drawContours(annotated_image, [box_points], 0, (255, 0, 0), 2)

    # Add text annotations
    if "area_cm2" in measurements:
        # Real-world measurements
        text_area = f"Area: {measurements['area_cm2']:.2f} cm²"
        text_size = f"Size: {measurements['width_cm']:.2f} x {measurements['height_cm']:.2f} cm"
    else:
        # Pixel measurements
        text_area = f"Area: {measurements['area_px']:.0f} px"
        text_size = f"Size: {measurements['width_px']:.0f} x {measurements['height_px']:.0f} px"

    cv2.putText(annotated_image, text_area, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(annotated_image, text_size, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return annotated_image
