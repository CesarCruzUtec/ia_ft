"""
YOLO Model Inspector
Displays detailed information about YOLO .pt model files.

Usage:
    python -m manual_tests.inspect_model
"""

from pathlib import Path

import torch

from config import MODELS_DIR


def inspect_model(model_path: Path):
    """Inspect a YOLO model file and display detailed information."""
    print("\n" + "=" * 70)
    print(f"📦 Model: {model_path.name}")
    print("=" * 70)

    try:
        # Load the model checkpoint
        print("\n⟳ Loading checkpoint...")
        # Note: weights_only=False is used because YOLO models contain custom classes.
        # Only use this with trusted model files.
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Basic file info
        print("\n📁 File Information:")
        print(f"  - Path: {model_path}")
        print(f"  - Size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")

        # Checkpoint structure
        print("\n🔑 Checkpoint Keys:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, dict):
                print(f"  - {key}: dict with {len(value)} items")
            elif isinstance(value, (list, tuple)):
                print(f"  - {key}: {type(value).__name__} with {len(value)} items")
            elif isinstance(value, torch.Tensor):
                print(f"  - {key}: Tensor {tuple(value.shape)}")
            else:
                print(f"  - {key}: {type(value).__name__}")

        # Model metadata
        if "model" in checkpoint:
            model_obj = checkpoint["model"]
            print("\n🤖 Model Object:")
            print(f"  - Type: {type(model_obj).__name__}")

            # Try to get model info
            if hasattr(model_obj, "yaml"):
                print("\n📋 Model Configuration (YAML):")
                yaml_data = model_obj.yaml
                if isinstance(yaml_data, dict):
                    for key, value in yaml_data.items():
                        if isinstance(value, (list, dict)):
                            print(f"  - {key}: {type(value).__name__} with {len(value)} items")
                        else:
                            print(f"  - {key}: {value}")
                else:
                    print(f"  {yaml_data}")

            if hasattr(model_obj, "names"):
                print(f"\n🏷️  Class Names ({len(model_obj.names)} classes):")
                names = model_obj.names
                if isinstance(names, dict):
                    for idx, name in names.items():
                        print(f"  {idx}: {name}")
                elif isinstance(names, list):
                    for idx, name in enumerate(names):
                        print(f"  {idx}: {name}")
                else:
                    print(f"  {names}")

            if hasattr(model_obj, "nc"):
                print(f"\n📊 Number of Classes: {model_obj.nc}")

            if hasattr(model_obj, "stride"):
                print(f"\n📐 Model Stride: {model_obj.stride}")

            if hasattr(model_obj, "pt_path"):
                print(f"\n💾 Original Path: {model_obj.pt_path}")

        # Training info
        if "epoch" in checkpoint:
            print("\n🎓 Training Info:")
            print(f"  - Epoch: {checkpoint['epoch']}")

            if "best_fitness" in checkpoint:
                print(f"  - Best Fitness: {checkpoint['best_fitness']}")

            if "train_args" in checkpoint or "training_args" in checkpoint:
                args = checkpoint.get("train_args") or checkpoint.get("training_args")
                if isinstance(args, dict):
                    print("\n  Training Arguments:")
                    for key, value in args.items():
                        if not key.startswith("_"):
                            print(f"    - {key}: {value}")

        # Optimizer info
        if "optimizer" in checkpoint:
            print(
                f"\n⚙️  Optimizer: {type(checkpoint['optimizer']).__name__ if isinstance(checkpoint['optimizer'], object) else 'Present'}"
            )

        # Version info
        if "version" in checkpoint:
            print(f"\n🔖 Version: {checkpoint['version']}")

        if "date" in checkpoint:
            print(f"📅 Date: {checkpoint['date']}")

        # EMA (Exponential Moving Average) info
        if "ema" in checkpoint:
            print("\n📈 EMA: Present")

        # Updates info
        if "updates" in checkpoint:
            print(f"🔄 Updates: {checkpoint['updates']}")

        print("\n" + "=" * 70)
        print("✅ Model inspection complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback

        traceback.print_exc()


def compare_models(model_paths: list[Path]):
    """Compare multiple models side by side."""
    print("\n" + "=" * 70)
    print("🔍 Model Comparison")
    print("=" * 70)

    models_info = []

    for model_path in model_paths:
        try:
            # Note: weights_only=False is used because YOLO models contain custom classes.
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            info = {
                "name": model_path.name,
                "size_mb": model_path.stat().st_size / (1024 * 1024),
                "epoch": checkpoint.get("epoch", "N/A"),
                "num_classes": None,
                "class_names": None,
            }

            if "model" in checkpoint:
                model_obj = checkpoint["model"]
                if hasattr(model_obj, "nc"):
                    info["num_classes"] = model_obj.nc
                if hasattr(model_obj, "names"):
                    info["class_names"] = model_obj.names

            models_info.append(info)

        except Exception as e:
            print(f"❌ Error loading {model_path.name}: {e}")

    # Display comparison table
    print("\n" + "-" * 70)
    print(f"{'Model':<40} {'Size (MB)':<12} {'Epoch':<8} {'Classes':<8}")
    print("-" * 70)

    for info in models_info:
        print(f"{info['name']:<40} {info['size_mb']:<12.2f} {str(info['epoch']):<8} {str(info['num_classes']):<8}")

    print("-" * 70)

    # Check for class name differences
    print("\n🏷️  Class Names Comparison:")
    for info in models_info:
        if info["class_names"]:
            print(f"\n{info['name']}:")
            names = info["class_names"]
            if isinstance(names, dict):
                print(f"  {', '.join(str(v) for v in names.values())}")
            elif isinstance(names, list):
                print(f"  {', '.join(names)}")

    print("\n" + "=" * 70)


def main():
    print("\n" + "=" * 70)
    print("🔍 YOLO Model Inspector")
    print("=" * 70)

    # List available models
    print("\n📦 Available models:")
    models = sorted(MODELS_DIR.glob("*.pt"))

    if not models:
        print("❌ No models found in models/ directory")
        print(f"   Path: {MODELS_DIR}")
        return

    for i, model in enumerate(models, 1):
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"  {i}. {model.name} ({size_mb:.2f} MB)")

    print("\nOptions:")
    print("  - Enter model number to inspect")
    print("  - Enter 'all' to compare all models")
    print("  - Enter 'c' to compare specific models")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "all":
        compare_models(models)
    elif choice == "c":
        print("\nEnter model numbers to compare (comma-separated):")
        indices_input = input("Models: ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in indices_input.split(",")]
            selected_models = [models[i] for i in indices if 0 <= i < len(models)]
            if selected_models:
                compare_models(selected_models)
            else:
                print("❌ No valid models selected")
        except (ValueError, IndexError):
            print("❌ Invalid input")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                inspect_model(models[idx])
            else:
                print("❌ Invalid model number")
        except ValueError:
            print("❌ Invalid input")


if __name__ == "__main__":
    main()
