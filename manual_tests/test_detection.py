"""
Manual test script for YOLO Detection.
Run this to test object detection on your own images.

Usage:
    python manual_tests/test_detection.py
"""

from config import MODELS_DIR
from core import DeviceManager, ImageManager
from modules.detection import YOLODetector


def list_available_models():
    """List available YOLO models."""
    models = list(MODELS_DIR.glob("*.pt"))
    return [m.stem for m in models]


def main():
    print("\n" + "=" * 70)
    print("🎯 Manual Test: YOLO Detection")
    print("=" * 70)

    # Initialize components
    print("\n1. Initializing components...")
    device_manager = DeviceManager()
    image_manager = ImageManager()
    detector = YOLODetector(device_manager)
    print(f"✓ Device: {device_manager.device}")

    # Show available models
    print("\n2. Available YOLO models:")
    models = list_available_models()
    if not models:
        print("❌ No models found in models/ directory")
        print(f"   Please add .pt files to: {MODELS_DIR}")
        return

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
    print("\n3. Select image to analyze")
    print("   You can provide:")
    print("     - Filename (e.g., 'test.jpg')")
    print("     - Relative path (e.g., 'mayor/brotes/image.jpg')")
    print("     - Absolute path")

    image_path = input("\nEnter image path: ").strip()
    if not image_path:
        print("❌ No path provided")
        return

    try:
        # Load image
        print("\n4. Loading image...")
        image = image_manager.load_image(image_path)
        print(f"✓ Image loaded: {image.shape}")

        # Run detection
        print(f"\n5. Running YOLO detection with model: {model_name}")
        detections = detector.detect_objects(image, model_name)

        # Display results
        print(f"\n{'=' * 70}")
        print("📊 Detection Results")
        print(f"{'=' * 70}")
        print(f"\n✓ Detected {len(detections)} object(s)\n")

        if detections:
            for i, det in enumerate(detections, 1):
                print(f"Object {i}:")
                print(f"  - Label: {det.label}")
                print(f"  - Confidence: {det.confidence:.2%}")
                print(f"  - Bounding Box: ({det.x1:.0f}, {det.y1:.0f}) → ({det.x2:.0f}, {det.y2:.0f})")
                print(f"  - Size: {det.x2 - det.x1:.0f}x{det.y2 - det.y1:.0f} pixels")
                print()
        else:
            print("No objects detected in the image.")

        print("=" * 70)
        print("✅ Detection test complete!")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
