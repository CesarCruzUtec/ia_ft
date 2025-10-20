from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def mask_to_binary(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Convert mask to binary (0 or 255).

    Args:
        mask: Input mask (can be 2D, 3D, tensor, etc.)
        threshold: Binarization threshold

    Returns:
        Binary mask (uint8)
    """
    # Handle torch tensors
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()

    # Convert to numpy if needed
    mask = np.array(mask)

    # Handle 3D masks (take first channel)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]

    # Normalize to 0-255
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # Binarize
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary


def measure_mask(mask: np.ndarray, px_per_cm: Optional[float] = None) -> Dict:
    """
    Measure mask dimensions in pixels and optionally in cm.

    Args:
        mask: Binary mask (uint8)
        px_per_cm: Scale factor (pixels per cm). If None, only pixel measurements returned.

    Returns:
        Dict with measurements: area_px, area_cm2, perimeter_px, perimeter_cm,
        width_px, height_px, width_cm, height_cm, angle
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No contours found in mask")

    # Use largest contour
    contour = max(contours, key=cv2.contourArea)

    # Pixel measurements
    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)

    # Minimum area rectangle
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    # Ensure width >= height
    if w < h:
        w, h = h, w
        angle = (angle + 90) % 180

    result = {
        "area_px": float(area_px),
        "perimeter_px": float(perimeter_px),
        "width_px": float(w),
        "height_px": float(h),
        "angle": float(angle),
        "center_x": float(cx),
        "center_y": float(cy),
        "contour_points": len(contour),
    }

    # Convert to cm if scale provided
    if px_per_cm is not None:
        result["area_cm2"] = area_px / (px_per_cm**2)
        result["perimeter_cm"] = perimeter_px / px_per_cm
        result["width_cm"] = w / px_per_cm
        result["height_cm"] = h / px_per_cm
        result["px_per_cm"] = px_per_cm

    return result


def draw_measurements(
    image: np.ndarray, mask: np.ndarray, measurements: Dict, color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw measurements on image.

    Args:
        image: Input image
        mask: Binary mask
        measurements: Dict from measure_mask()
        color: Color for drawing (BGR)

    Returns:
        Image with annotations
    """
    img = image.copy()

    # Draw contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, [contour], -1, color, 2)

    # Draw bounding box
    cx, cy = measurements["center_x"], measurements["center_y"]
    w, h = measurements["width_px"], measurements["height_px"]
    angle = measurements["angle"]

    box = cv2.boxPoints(((cx, cy), (w, h), angle))
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

    # Add text
    if "area_cm2" in measurements:
        text = f"Area: {measurements['area_cm2']:.2f} cm²"
        text2 = f"Size: {measurements['width_cm']:.2f} x {measurements['height_cm']:.2f} cm"
    else:
        text = f"Area: {measurements['area_px']:.0f} px"
        text2 = f"Size: {measurements['width_px']:.0f} x {measurements['height_px']:.0f} px"

    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return img
