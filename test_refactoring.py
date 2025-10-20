#!/usr/bin/env python3
"""
Quick test script to verify the refactored backend structure.
Run this to check if all imports work correctly.
"""


def test_imports():
    """Test that all modules can be imported without errors."""
    print("=" * 60)
    print("Testing Backend Refactoring - Import Check")
    print("=" * 60)

    try:
        print("\n✓ Testing config...")
        import config

        print(f"  - SAM2 models: {len(config.SAM2_MODEL_CONFIGS)}")
        print(f"  - Models dir: {config.MODELS_DIR}")

        print("\n✓ Testing core...")
        from core import DeviceManager, ImageManager

        print(f"  - DeviceManager: {DeviceManager}")
        print(f"  - ImageManager: {ImageManager}")

        print("\n✓ Testing modules.detection...")
        from modules.detection import DetectionBox, YOLODetector

        print(f"  - YOLODetector: {YOLODetector}")
        print(f"  - DetectionBox: {DetectionBox}")

        print("\n✓ Testing modules.segmentation...")
        from modules.segmentation import SAM2Segmentor, SegmentationBox, mask_to_base64

        print(f"  - SAM2Segmentor: {SAM2Segmentor}")
        print(f"  - SegmentationBox: {SegmentationBox}")
        print(f"  - mask_to_base64: {mask_to_base64}")

        print("\n✓ Testing modules.measurement...")
        from modules.measurement import ArucoDetector, ArucoMeasurer, MeasurementBox

        print(f"  - ArucoMeasurer: {ArucoMeasurer}")
        print(f"  - MeasurementBox: {MeasurementBox}")
        print(f"  - ArucoDetector: {ArucoDetector}")

        print("\n✓ Testing services...")
        from services import PipelineService

        print(f"  - PipelineService: {PipelineService}")

        print("\n✓ Testing utils...")
        from utils import print_dict

        print(f"  - print_dict: {print_dict}")

        print("\n" + "=" * 60)
        print("✅ ALL IMPORTS SUCCESSFUL!")
        print("=" * 60)
        print("\nThe refactored backend structure is working correctly.")
        print("You can now run the FastAPI server with:")
        print("  uvicorn main:app --reload")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ IMPORT FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_initialization():
    """Test that main components can be initialized."""
    print("\n" + "=" * 60)
    print("Testing Component Initialization")
    print("=" * 60)

    try:
        print("\n⟳ Initializing DeviceManager...")
        from core import DeviceManager

        device_mgr = DeviceManager()
        print(f"  Device: {device_mgr.device}")

        print("\n⟳ Initializing ImageManager...")
        from core import ImageManager

        img_mgr = ImageManager()
        print(f"  Images directory: {img_mgr.images_directory}")

        print("\n✅ INITIALIZATION SUCCESSFUL!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ INITIALIZATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    # Test imports
    imports_ok = test_imports()

    if imports_ok:
        # Test initialization
        init_ok = test_initialization()

        if init_ok:
            print("\n🎉 All tests passed! Backend refactoring is complete.")
            sys.exit(0)
        else:
            print("\n⚠ Initialization tests failed.")
            sys.exit(1)
    else:
        print("\n⚠ Import tests failed.")
        sys.exit(1)
