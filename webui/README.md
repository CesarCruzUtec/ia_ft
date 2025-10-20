# Gradio WebUI for Local Testing

This folder contains a Gradio-based interface for testing the image analysis pipeline locally.

## Usage

1. Make sure dependencies are installed (from project root):
   ```fish
   uv sync
   ```

2. Run the app as a Python module (from project root):
   ```fish
   uv run python -m webui.app
   ```
   
   Or without uv:
   ```fish
   python -m webui.app
   ```

3. Open your browser at http://localhost:7860

## Features

- 📤 **Upload Images:** Upload an image directly from your browser
- 🎯 **YOLO Detection:** Automatically discovers and lists all `.pt` models in `models/` directory
- ✂️ **SAM2 Segmentation:** Dynamically loads available SAM2 models from `config.py`
- 📏 **ArUco Measurement:** Specify ArUco marker size for real-world measurements
- 🖼️ **Visual Results:** View detection bounding boxes and segmentation mask overlays
- 📊 **Detailed Results:** See step-by-step pipeline output with all metrics
- 🖥️ **System Monitor:** Real-time CPU, RAM, and GPU utilization monitoring
  - Updates every second
  - Shows CPU percentage, RAM usage, and GPU metrics (memory, utilization, temperature)
  - NVIDIA GPU support via `nvidia-ml-py`

## Model Discovery

The WebUI automatically discovers available models:

- **YOLO Models:** Scans `models/` directory for `.pt` files
  - Add new YOLO models by placing `.pt` files in the `models/` folder
  - Model names are derived from filenames (without `.pt` extension)
  - Example: `my_custom_model.pt` → appears as `my_custom_model` in dropdown

- **SAM2 Models:** Reads from `config.py` `SAM2_MODEL_CONFIGS`
  - Currently available: tiny, small, base_plus, large
  - To add custom SAM2 models, edit `config.py`

The dropdowns show how many models were found (e.g., "2 model(s) found").

---

**Note:** This UI is for local testing only. For production, use the API endpoints.
