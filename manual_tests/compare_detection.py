"""
Compare Detection Results Between Two Models
Tests both models on the same image to diagnose differences.

Usage:
    python -m manual_tests.compare_detection
"""

from pathlib import Path

from ultralytics import YOLO

from config import IMAGES_DIR, MODELS_DIR
from core.image_manager import ImageManager


def compare_models_detection():
    """Compare detection results between two models on the same image."""
    print("\n" + "=" * 70)
    print("🔍 Model Detection Comparison")
    print("=" * 70)

    # List available models
    models_dir = Path(MODELS_DIR)
    model_files = sorted(models_dir.glob("*.pt"))

    print("\n📦 Available models:")
    for idx, model_file in enumerate(model_files, 1):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  {idx}. {model_file.name} ({size_mb:.2f} MB)")

    # Select two models to compare
    print("\nSelect first model:")
    choice1 = input("Model number: ").strip()
    try:
        model1_path = model_files[int(choice1) - 1]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return

    print("\nSelect second model:")
    choice2 = input("Model number: ").strip()
    try:
        model2_path = model_files[int(choice2) - 1]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return

    # Select test image
    print("\n📁 Enter path to test image:")
    print(f"   (Images directory: {IMAGES_DIR})")
    image_path = input("Image path or name: ").strip()

    try:
        img_manager = ImageManager()
        image = img_manager.load_image(image_path)
        print(f"✅ Loaded image: {image.shape}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Test Model 1
    print("\n" + "=" * 70)
    print(f"🔍 Testing Model 1: {model1_path.name}")
    print("=" * 70)

    try:
        print("⟳ Loading model...")
        model1 = YOLO(str(model1_path))

        print("✅ Model loaded")
        print(f"   Class names: {model1.names}")
        print(f"   Device: {model1.device}")

        print("\n⟳ Running inference...")
        results1 = model1(image, verbose=False)

        print("✅ Inference complete")
        print(f"   Detections: {len(results1[0].boxes)}")

        if len(results1[0].boxes) > 0:
            print("\n   Detection details:")
            for idx, box in enumerate(results1[0].boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = model1.names[cls]
                xyxy = box.xyxy[0].tolist()
                print(f"     {idx + 1}. {cls_name} (conf: {conf:.3f}) at {[f'{x:.1f}' for x in xyxy]}")
        else:
            print("   ⚠️  No detections found")

    except Exception as e:
        print(f"❌ Error with Model 1: {e}")
        import traceback

        traceback.print_exc()

    # Test Model 2
    print("\n" + "=" * 70)
    print(f"🔍 Testing Model 2: {model2_path.name}")
    print("=" * 70)

    try:
        print("⟳ Loading model...")
        model2 = YOLO(str(model2_path))

        print("✅ Model loaded")
        print(f"   Class names: {model2.names}")
        print(f"   Device: {model2.device}")

        print("\n⟳ Running inference...")
        results2 = model2(image, verbose=False)

        print("✅ Inference complete")
        print(f"   Detections: {len(results2[0].boxes)}")

        if len(results2[0].boxes) > 0:
            print("\n   Detection details:")
            for idx, box in enumerate(results2[0].boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = model2.names[cls]
                xyxy = box.xyxy[0].tolist()
                print(f"     {idx + 1}. {cls_name} (conf: {conf:.3f}) at {[f'{x:.1f}' for x in xyxy]}")
        else:
            print("   ⚠️  No detections found")

    except Exception as e:
        print(f"❌ Error with Model 2: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print(f"Model 1 ({model1_path.name}): {len(results1[0].boxes) if 'results1' in locals() else 'ERROR'} detections")
    print(f"Model 2 ({model2_path.name}): {len(results2[0].boxes) if 'results2' in locals() else 'ERROR'} detections")
    print("=" * 70)


if __name__ == "__main__":
    compare_models_detection()
