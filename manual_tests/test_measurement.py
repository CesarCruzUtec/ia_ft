"""
Manual test script for ArUco Measurement.
Run this to test measurement on your own images with ArUco markers.

Usage:
    python manual_tests/test_measurement.py
"""

from core import ImageManager
from modules.measurement import ArucoMeasurer


def main():
    print("\n" + "=" * 70)
    print("📏 Manual Test: ArUco Measurement")
    print("=" * 70)

    # Initialize components
    print("\n1. Initializing components...")
    image_manager = ImageManager()
    measurer = ArucoMeasurer()
    print("✓ ArUco measurer initialized")

    # Get image path
    print("\n2. Select image with ArUco markers")
    print("   (Image should contain ArUco markers for scale reference)")
    image_path = input("Enter image path: ").strip()
    if not image_path:
        print("❌ No path provided")
        return

    try:
        # Load image
        print("\n3. Loading image...")
        image = image_manager.load_image(image_path)
        print(f"✓ Image loaded: {image.shape}")

        # Get marker size
        print("\n4. ArUco marker configuration")
        marker_size_input = input("Enter marker size in cm (default: 4.9): ").strip()
        marker_size_cm = float(marker_size_input) if marker_size_input else 4.9
        print(f"✓ Using marker size: {marker_size_cm} cm")

        # Get segmentation masks (boxes with masks)
        print("\n5. Define objects to measure")
        print("   For this test, we'll create sample boxes.")
        print("   In production, these come from the segmentation step.")

        use_sample = input("Use sample box? (y/n, default: y): ").strip().lower()

        boxes = []
        if use_sample != "n":
            # Create a sample box in the center
            h, w = image.shape[:2]
            # Create a simple mask (you'd normally get this from SAM2)
            import numpy as np

            from modules.segmentation.mask_utils import mask_to_base64

            mask = np.zeros((1, h, w), dtype=bool)
            mask[0, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
            mask_base64 = mask_to_base64(mask)

            boxes = [
                {
                    "x1": w // 4,
                    "y1": h // 4,
                    "x2": 3 * w // 4,
                    "y2": 3 * h // 4,
                    "label": "sample_object",
                    "confidence": 1.0,
                    "mask_base64": mask_base64,
                    "mask_score": 0.95,
                }
            ]
            print("✓ Using sample box in image center")
        else:
            print("❌ Manual box creation not implemented in this test")
            print("   Run the full pipeline to test with real segmentation masks")
            return

        # Run measurement
        print("\n6. Running ArUco measurement...")
        measured, markers_count, scale = measurer.measure_objects(
            image=image, boxes=boxes, marker_size_cm=marker_size_cm, marker_id=None
        )

        # Display results
        print("\n{'=' * 70}")
        print("📊 Measurement Results")
        print(f"{'=' * 70}")

        if markers_count == 0:
            print("\n⚠️  No ArUco markers detected!")
            print("   - Make sure your image contains ArUco markers")
            print("   - Markers should be clearly visible")
            print("   - Using dictionary: DICT_4X4_50")
            print("\n✓ Scale: N/A (no markers)")
            print("✓ Objects measured: 0")
        else:
            print(f"\n✓ ArUco markers detected: {markers_count}")
            print(f"✓ Scale: {scale:.2f} pixels/cm")
            print(f"✓ Objects measured: {len(measured)}\n")

            for i, meas in enumerate(measured, 1):
                print(f"Measurement {i}:")
                print(f"  - Label: {meas.label}")
                if meas.area_cm2:
                    print(f"  - Area: {meas.area_cm2:.2f} cm²")
                    print(f"  - Perimeter: {meas.perimeter_cm:.2f} cm")
                    print(f"  - Width: {meas.width_cm:.2f} cm")
                    print(f"  - Height: {meas.height_cm:.2f} cm")
                    print(f"  - Angle: {meas.angle:.2f}°")
                else:
                    print("  - No measurements (marker detection failed)")
                print()

        print("=" * 70)
        print("✅ Measurement test complete!")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
