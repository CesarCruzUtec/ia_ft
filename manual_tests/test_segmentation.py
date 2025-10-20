"""
Manual test script for SAM2 Segmentation.
Run this to test segmentation on your own images and bounding boxes.

Usage:
    python manual_tests/test_segmentation.py
"""

from config import SAM2_MODEL_CONFIGS
from core import DeviceManager, ImageManager
from modules.segmentation import SAM2Segmentor


def main():
    print("\n" + "=" * 70)
    print("✂️  Manual Test: SAM2 Segmentation")
    print("=" * 70)

    # Initialize components
    print("\n1. Initializing components...")
    device_manager = DeviceManager()
    image_manager = ImageManager()
    segmentor = SAM2Segmentor(device_manager)
    print(f"✓ Device: {device_manager.device}")

    # Show available SAM2 models
    print("\n2. Available SAM2 models:")
    models = list(SAM2_MODEL_CONFIGS.keys())
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")

    # Get model selection
    model_choice = input(f"\nSelect model (1-{len(models)}): ").strip()
    try:
        model_idx = int(model_choice) - 1
        model_name = models[model_idx]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return

    # Get image path
    print("\n3. Select image to segment")
    image_path = input("Enter image path: ").strip()
    if not image_path:
        print("❌ No path provided")
        return

    try:
        # Load image
        print("\n4. Loading image...")
        image = image_manager.load_image(image_path)
        print(f"✓ Image loaded: {image.shape}")

        # Get bounding boxes
        print("\n5. Define bounding boxes")
        print("   Enter boxes in format: x1,y1,x2,y2,label")
        print("   Example: 100,100,200,200,object")
        print("   Enter empty line when done")

        boxes = []
        box_num = 1
        while True:
            box_input = input(f"   Box {box_num}: ").strip()
            if not box_input:
                break

            try:
                parts = box_input.split(",")
                x1, y1, x2, y2 = map(int, parts[:4])
                label = parts[4] if len(parts) > 4 else f"object{box_num}"

                boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label, "confidence": 1.0})
                print(f"   ✓ Added box: {label} at ({x1},{y1})-({x2},{y2})")
                box_num += 1
            except (ValueError, IndexError):
                print("   ❌ Invalid format. Use: x1,y1,x2,y2,label")

        if not boxes:
            print("\n❌ No boxes defined. Creating a default box...")
            # Create a box in the center of the image
            h, w = image.shape[:2]
            boxes = [
                {"x1": w // 4, "y1": h // 4, "x2": 3 * w // 4, "y2": 3 * h // 4, "label": "default", "confidence": 1.0}
            ]
            print(f"   Using default box: {boxes[0]}")

        # Run segmentation
        print(f"\n6. Running SAM2 segmentation with model: {model_name}")
        segmented = segmentor.segment_boxes(image, boxes, model_name)

        # Display results
        print(f"\n{'=' * 70}")
        print("📊 Segmentation Results")
        print(f"{'=' * 70}")
        print(f"\n✓ Generated {len(segmented)} mask(s)\n")

        for i, seg in enumerate(segmented, 1):
            print(f"Mask {i}:")
            print(f"  - Label: {seg.label}")
            print(f"  - Confidence: {seg.confidence:.2%}")
            print(f"  - Mask Score: {seg.mask_score:.2%}")
            print(f"  - Bounding Box: ({seg.x1:.0f}, {seg.y1:.0f}) → ({seg.x2:.0f}, {seg.y2:.0f})")
            print(f"  - Mask Base64 length: {len(seg.mask_base64)} chars")
            print()

        print("=" * 70)
        print("✅ Segmentation test complete!")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
