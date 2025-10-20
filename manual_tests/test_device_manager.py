"""
Manual test script for DeviceManager.
Run this to test device detection and management.
"""

from core import DeviceManager


def main():
    print("\n" + "=" * 70)
    print("🔧 Manual Test: DeviceManager")
    print("=" * 70)

    # Initialize DeviceManager
    print("\n1. Initializing DeviceManager...")
    device_manager = DeviceManager()

    # Display device info
    print(f"\n✓ Device detected: {device_manager.device}")
    print(f"✓ Device index: {device_manager.device.index}")
    print(f"✓ Device type: {device_manager.device.type}")

    # Test cache clearing
    print("\n2. Testing cache clear...")
    device_manager.clear_cache()
    print("✓ Cache cleared successfully")

    print("\n" + "=" * 70)
    print("✅ DeviceManager test complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
