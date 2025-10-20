#!/usr/bin/env python3
"""
Main script to measure object dimensions using ArUco markers and segmentation masks.

Usage:
    python main.py --image path/to/image.jpg --mask path/to/mask.png --marker-size 4.9
"""

import argparse

import cv2
import numpy as np
from src.detector import ArucoDetector
from src.utils import draw_measurements, mask_to_binary, measure_mask


def main():
    parser = argparse.ArgumentParser(description="Measure objects using ArUco markers")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, required=True, help="Path to segmentation mask")
    parser.add_argument("--marker-size", type=float, required=True, help="ArUco marker size in cm")
    parser.add_argument("--marker-id", type=int, default=None, help="Specific marker ID to use")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--show", action="store_true", help="Display result")

    args = parser.parse_args()

    # Initialize detector
    print("Initializing ArUco detector...")
    detector = ArucoDetector()

    # Load image
    print(f"Loading image: {args.image}")
    image = detector.load_image(args.image)

    # Detect markers
    print("Detecting ArUco markers...")
    _, ids = detector.detect_markers()

    if ids is None or len(ids) == 0:
        print("ERROR: No ArUco markers detected!")
        return

    print(f"Found {len(ids)} marker(s): {ids.flatten().tolist()}")

    # Calculate scale
    print(f"Calculating scale (marker size: {args.marker_size} cm)...")
    px_per_cm = detector.calculate_scale(args.marker_size, args.marker_id)
    print(f"Scale: {px_per_cm:.2f} pixels/cm")

    # Load and process mask
    print(f"Loading mask: {args.mask}")
    mask_raw = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        # Try loading as numpy
        mask_raw = np.load(args.mask)

    mask = mask_to_binary(mask_raw)
    print(f"Mask shape: {mask.shape}")

    # Measure
    print("Measuring object...")
    measurements = measure_mask(mask, px_per_cm)

    # Print results
    print("\n" + "=" * 50)
    print("MEASUREMENTS")
    print("=" * 50)
    print(f"Area:      {measurements['area_cm2']:.2f} cm²  ({measurements['area_px']:.0f} px)")
    print(f"Perimeter: {measurements['perimeter_cm']:.2f} cm  ({measurements['perimeter_px']:.0f} px)")
    print(f"Width:     {measurements['width_cm']:.2f} cm  ({measurements['width_px']:.0f} px)")
    print(f"Height:    {measurements['height_cm']:.2f} cm  ({measurements['height_px']:.0f} px)")
    print(f"Angle:     {measurements['angle']:.1f}°")
    print("=" * 50 + "\n")

    # Draw results
    result_img = detector.draw_markers(image)
    result_img = draw_measurements(result_img, mask, measurements)

    # Save output
    if args.output:
        cv2.imwrite(args.output, result_img)
        print(f"Result saved to: {args.output}")

    # Show
    if args.show:
        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
