"""
Manual test script for ImageManager.
Run this to test loading images from your local files.

Usage:
    python manual_tests/test_image_manager.py

Then enter the path to your test image when prompted.
"""

from pathlib import Path

from core import ImageManager


def main():
    print("\n" + "=" * 70)
    print("🖼️  Manual Test: ImageManager")
    print("=" * 70)

    # Initialize ImageManager
    print("\n1. Initializing ImageManager...")
    print("   Using images directory: images/")
    image_manager = ImageManager()

    # Get image path from user
    print("\n2. Load an image")
    print("\nYou can provide:")
    print("  - Filename only (e.g., 'test.jpg') - will search in images/")
    print("  - Relative path (e.g., 'mayor/brotes/image.jpg')")
    print("  - Absolute path (e.g., 'C:\\path\\to\\image.jpg')")

    image_path = input("\nEnter image path: ").strip()

    if not image_path:
        print("❌ No path provided. Exiting.")
        return

    try:
        # Load the image
        print(f"\n3. Loading image: {image_path}")
        image = image_manager.load_image(image_path)

        print("\n✓ Image loaded successfully!")
        print(f"  - Shape: {image.shape}")
        print(f"  - Size: {image.shape[1]}x{image.shape[0]} pixels")
        print(f"  - Channels: {image.shape[2]}")
        print(f"  - Data type: {image.dtype}")
        print(f"  - Memory: {image.nbytes / 1024:.2f} KB")

        # Test caching
        print("\n4. Testing image cache...")
        current_source = image_manager.get_current_image_source()
        print(f"  - Cached image source: {current_source}")
        print(f"  - Is cached: {image_manager.is_image_cached(image_path)}")

        # Load again (should use cache)
        print("\n5. Loading same image again (should use cache)...")
        _ = image_manager.load_image(image_path)
        print("✓ Image loaded from cache")

        # Clear cache
        print("\n6. Clearing cache...")
        image_manager.clear_cache()
        print("✓ Cache cleared")

        print("\n" + "=" * 70)
        print("✅ ImageManager test complete!")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTip: Make sure the image exists in the images/ directory")
        print("     or provide the full path to the image.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
