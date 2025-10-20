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
- 🎯 **YOLO Detection:** Select from available YOLO models
- ✂️ **SAM2 Segmentation:** Choose SAM2 model for precise segmentation
- 📏 **ArUco Measurement:** Specify ArUco marker size for real-world measurements
- 🖼️ **Visual Results:** View detection bounding boxes and segmentation mask overlays
- 📊 **Detailed Results:** See step-by-step pipeline output with all metrics
- 🖥️ **System Monitor:** Real-time CPU, RAM, and GPU utilization monitoring
  - Updates every second
  - Shows CPU percentage, RAM usage, and GPU metrics (memory, utilization, temperature)
  - NVIDIA GPU support via `nvidia-ml-py`

---

**Note:** This UI is for local testing only. For production, use the API endpoints.
