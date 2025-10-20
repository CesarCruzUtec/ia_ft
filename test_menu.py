#!/usr/bin/env python3
"""
Launcher for manual testing scripts.
Provides a menu to select which module to test.
"""

import subprocess
import sys
from pathlib import Path


def show_menu():
    """Display the test menu."""
    print("\n" + "=" * 70)
    print("🧪 Manual Testing Menu")
    print("=" * 70)
    print("\nSelect a module to test:\n")
    print("  1. DeviceManager        - Test CPU/GPU detection")
    print("  2. ImageManager         - Test image loading")
    print("  3. Detection (YOLO)     - Test object detection")
    print("  4. Segmentation (SAM2)  - Test segmentation")
    print("  5. Measurement (ArUco)  - Test measurements")
    print("  6. Full Pipeline        - Test complete workflow")
    print("\n  0. Exit")
    print("\n" + "=" * 70)


def run_test(choice):
    """Run the selected test."""
    tests = {
        "1": ("manual_tests.test_device_manager", "DeviceManager"),
        "2": ("manual_tests.test_image_manager", "ImageManager"),
        "3": ("manual_tests.test_detection", "Detection"),
        "4": ("manual_tests.test_segmentation", "Segmentation"),
        "5": ("manual_tests.test_measurement", "Measurement"),
        "6": ("manual_tests.test_full_pipeline", "Full Pipeline"),
    }

    if choice not in tests:
        print("\n❌ Invalid choice. Please try again.")
        return False

    module_name, name = tests[choice]

    print(f"\n{'=' * 70}")
    print(f"Running: {name} Test")
    print(f"{'=' * 70}\n")

    try:
        # Run as module so package imports like `from core import ...` work
        result = subprocess.run([sys.executable, "-m", module_name])
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        return False


def main():
    """Main menu loop."""
    print("\n🥔 Image Analysis Pipeline - Manual Testing")
    print("=" * 70)
    print("This tool helps you test individual modules interactively.")

    while True:
        show_menu()

        try:
            choice = input("\nYour choice (0-6): ").strip()

            if choice == "0":
                print("\n👋 Goodbye!\n")
                break

            run_test(choice)

            input("\n Press Enter to continue...")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
