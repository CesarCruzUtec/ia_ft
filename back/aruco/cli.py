"""Simple test script for quick experimentation."""

import cv2
from src.detector import ArucoDetector
from src.utils import draw_measurements, mask_to_binary, measure_mask

# Example usage
if __name__ == "__main__":
    # Configuration
    IMAGE_PATH = "../images/test_image.jpg"
    MASK_PATH = "../images/test_mask.png"
    MARKER_SIZE_CM = 4.9

    # Initialize
    detector = ArucoDetector()

    # Process
    image = detector.load_image(IMAGE_PATH)
    corners, ids = detector.detect_markers()

    print(f"Detected markers: {ids.flatten().tolist() if ids is not None else []}")

    if ids is not None:
        px_per_cm = detector.calculate_scale(MARKER_SIZE_CM)
        print(f"Scale: {px_per_cm:.2f} px/cm")

        # Load mask
        mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
        mask = mask_to_binary(mask)

        # Measure
        measurements = measure_mask(mask, px_per_cm)
        print(f"Area: {measurements['area_cm2']:.2f} cm²")
        print(f"Dimensions: {measurements['width_cm']:.2f} x {measurements['height_cm']:.2f} cm")

        # Visualize
        result = draw_measurements(image, mask, measurements)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
