"""
Utility functions for mask processing and conversion.
"""

import base64

import cv2
import numpy as np

from config import MASK_ALPHA, MASK_BORDER_COLOR, MASK_BORDER_THICKNESS, MASK_COLOR_RGB


def mask_to_base64(mask: np.ndarray) -> str:
    """
    Convert a binary mask to a base64-encoded RGBA PNG image.

    The mask is visualized with a semi-transparent overlay and white borders.

    Args:
        mask: Binary mask array (2D or 3D numpy array)

    Returns:
        Base64-encoded data URI string (data:image/png;base64,...)
    """
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)

    # Handle 3D masks by taking the first channel
    if mask.ndim == 3:
        mask = mask[0]

    height, width = mask.shape

    # Create RGBA image directly as uint8 to save memory
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

    # Set colored overlay where mask is True
    mask_indices = mask > 0
    rgba_image[mask_indices, 0] = MASK_COLOR_RGB[0]  # Red channel
    rgba_image[mask_indices, 1] = MASK_COLOR_RGB[1]  # Green channel
    rgba_image[mask_indices, 2] = MASK_COLOR_RGB[2]  # Blue channel
    rgba_image[mask_indices, 3] = int(MASK_ALPHA * 255)  # Alpha channel

    # Draw white borders around the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Smooth contours
    smoothed_contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]

    rgba_image = cv2.drawContours(
        rgba_image, smoothed_contours, contourIdx=-1, color=MASK_BORDER_COLOR, thickness=MASK_BORDER_THICKNESS
    )

    # Encode to PNG using OpenCV
    # cv2.imencode expects BGRA for 4-channel images, so convert RGBA to BGRA
    bgra_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
    success, buffer = cv2.imencode(".png", bgra_image)

    if not success:
        raise ValueError("Failed to encode mask to PNG")

    # Convert to base64
    base64_bytes = base64.b64encode(buffer)
    base64_string = base64_bytes.decode("utf-8")

    return f"data:image/png;base64,{base64_string}"


def base64_to_mask(mask_base64: str) -> np.ndarray:
    """
    Convert a base64-encoded mask image back to a binary numpy array.

    Args:
        mask_base64: Base64-encoded data URI string

    Returns:
        Binary mask array (uint8, values 0 or 1)
    """
    # Extract base64 data from data URI
    if "," in mask_base64:
        _, encoded_data = mask_base64.split(",", 1)
    else:
        encoded_data = mask_base64

    # Decode base64
    mask_bytes = base64.b64decode(encoded_data)

    # Decode PNG using OpenCV
    nparr = np.frombuffer(mask_bytes, np.uint8)
    mask_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Convert to grayscale if needed
    if len(mask_image.shape) == 3:
        if mask_image.shape[2] == 4:
            # BGRA - use alpha channel or convert to grayscale
            mask_array = cv2.cvtColor(mask_image, cv2.COLOR_BGRA2GRAY)
        else:
            # BGR
            mask_array = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_array = mask_image

    # Binarize
    binary_mask = (mask_array > 128).astype(np.uint8)

    return binary_mask
