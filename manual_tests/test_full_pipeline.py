"""
Manual test script for the complete pipeline.
Run this to test the entire workflow: Detection → Segmentation → Measurement

Usage:
    python manual_tests/test_full_pipeline.py
"""

from config import MODELS_DIR, SAM2_MODEL_CONFIGS
from services import PipelineService


def list_yolo_models():
    """List available YOLO models."""
    models = list(MODELS_DIR.glob("*.pt"))
    return [m.stem for m in models]


def main():
    print("\n" + "=" * 70)
    print("🚀 Manual Test: Full Pipeline")
    print("=" * 70)

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = PipelineService()
    status = pipeline.get_status()
    print(f"✓ Device: {status['device']}")

    # Select YOLO model
    print("\n2. Available YOLO models:")
    yolo_models = list_yolo_models()
    if not yolo_models:
        print("❌ No YOLO models found in models/ directory")
        return

    for i, model in enumerate(yolo_models, 1):
        print(f"   {i}. {model}")

    yolo_choice = input(f"\nSelect YOLO model (1-{len(yolo_models)}): ").strip()
    try:
        yolo_idx = int(yolo_choice) - 1
        yolo_model = yolo_models[yolo_idx]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return

    # Select SAM2 model
    print("\n3. Available SAM2 models:")
    sam2_models = list(SAM2_MODEL_CONFIGS.keys())
    for i, model in enumerate(sam2_models, 1):
        print(f"   {i}. {model}")

    sam2_choice = input(f"\nSelect SAM2 model (1-{len(sam2_models)}): ").strip()
    try:
        sam2_idx = int(sam2_choice) - 1
        sam2_model = sam2_models[sam2_idx]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return

    # Get image
    print("\n4. Select image to analyze")
    image_path = input("Enter image path: ").strip()
    if not image_path:
        print("❌ No path provided")
        return

    # Get marker size
    marker_input = input("\n5. Enter ArUco marker size in cm (default: 4.9): ").strip()
    marker_size = float(marker_input) if marker_input else 4.9

    try:
        # Extract image name
        from pathlib import Path

        image_name = Path(image_path).name

        # STEP 1: Detection
        print(f"\n{'=' * 70}")
        print("STEP 1: Object Detection")
        print(f"{'=' * 70}")
        detections = pipeline.detect_objects(model_name=yolo_model, image_name=image_name)

        print(f"\n✓ Detected {len(detections)} object(s)")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det.label} ({det.confidence:.2%})")

        if not detections:
            print("\n❌ No objects detected. Pipeline stopped.")
            return

        # Convert to dict for next step
        boxes = [d.model_dump() for d in detections]

        # STEP 2: Segmentation
        print(f"\n{'=' * 70}")
        print("STEP 2: Segmentation")
        print(f"{'=' * 70}")
        segmented = pipeline.segment_objects(model_name=sam2_model, image_name=image_path, boxes=boxes)

        print(f"\n✓ Generated {len(segmented)} mask(s)")
        for i, seg in enumerate(segmented, 1):
            print(f"  {i}. {seg.label} (mask score: {seg.mask_score:.2%})")

        # Convert to dict for next step
        masks = [s.model_dump() for s in segmented]

        # STEP 3: Measurement
        print(f"\n{'=' * 70}")
        print("STEP 3: Measurement")
        print(f"{'=' * 70}")
        measured, markers_count, scale = pipeline.measure_objects(
            image_name=image_path, boxes=masks, marker_size_cm=marker_size, marker_id=None
        )

        if markers_count == 0:
            print("\n⚠️  No ArUco markers detected")
            print(f"✓ Objects processed: {len(masks)}")
            print("   (Measurements unavailable without markers)")
        else:
            print(f"\n✓ ArUco markers: {markers_count}")
            print(f"✓ Scale: {scale:.2f} px/cm")
            print(f"✓ Objects measured: {len(measured)}")

        # FINAL RESULTS
        print(f"\n{'=' * 70}")
        print("📊 FINAL RESULTS")
        print(f"{'=' * 70}")

        for i, meas in enumerate(measured if measured else segmented, 1):
            print(f"\nObject {i}: {meas.label}")
            print(f"  - Confidence: {meas.confidence:.2%}")
            print(f"  - Bounding Box: ({meas.x1:.0f}, {meas.y1:.0f}) → ({meas.x2:.0f}, {meas.y2:.0f})")

            if hasattr(meas, "mask_score"):
                print(f"  - Mask Score: {meas.mask_score:.2%}")

            if hasattr(meas, "area_cm2") and meas.area_cm2:
                print(f"  - Area: {meas.area_cm2:.2f} cm²")
                print(f"  - Perimeter: {meas.perimeter_cm:.2f} cm")
                print(f"  - Width: {meas.width_cm:.2f} cm")
                print(f"  - Height: {meas.height_cm:.2f} cm")
                print(f"  - Angle: {meas.angle:.2f}°")

        print("\n" + "=" * 70)
        print("✅ Full pipeline test complete!")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
