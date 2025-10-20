import base64
import io
from collections import deque
from datetime import datetime
from typing import Any, Dict

import gradio as gr
import pandas as pd
import psutil
from PIL import Image, ImageDraw

try:
    import pynvml  # nvidia-ml-py provides the pynvml module

    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

from config import MODELS_DIR, SAM2_MODEL_CONFIGS
from services.pipeline_service import PipelineService

# Initialize pipeline
pipeline = PipelineService()


def get_available_yolo_models():
    """Scan MODELS_DIR for .pt files and return model names (without extension)."""
    if not MODELS_DIR.exists():
        return []
    pt_files = list(MODELS_DIR.glob("*.pt"))
    # Return model names without .pt extension
    model_names = [f.stem for f in pt_files]
    return sorted(model_names) if model_names else []


def get_available_sam2_models():
    """Return available SAM2 model names from config."""
    return list(SAM2_MODEL_CONFIGS.keys())


# Dynamically load available models
YOLO_MODELS = get_available_yolo_models()
SAM2_MODELS = get_available_sam2_models()

# Historical data for graphs (store last 60 data points = 1 minute at 1s interval)
MAX_HISTORY = 60
history_timestamps = deque(maxlen=MAX_HISTORY)
history_cpu = deque(maxlen=MAX_HISTORY)
history_ram = deque(maxlen=MAX_HISTORY)
history_gpu_util = deque(maxlen=MAX_HISTORY)  # GPU 0 utilization
history_gpu_mem = deque(maxlen=MAX_HISTORY)  # GPU 0 memory %


def decode_mask(base64_str):
    """Decode base64 mask to PIL Image."""
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))


def overlay_mask_on_image(original_img, mask_img):
    """Overlay a semi-transparent mask on the original image."""
    # Ensure both images are RGBA
    if original_img.mode != "RGBA":
        original_img = original_img.convert("RGBA")
    if mask_img.mode != "RGBA":
        mask_img = mask_img.convert("RGBA")

    # Resize mask to match original image if needed
    if mask_img.size != original_img.size:
        mask_img = mask_img.resize(original_img.size, Image.Resampling.LANCZOS)

    # Composite the images
    result = Image.alpha_composite(original_img, mask_img)
    return result.convert("RGB")


def get_system_metrics() -> Dict[str, Any]:
    """Collect CPU, RAM, and GPU metrics (if available)."""
    metrics: Dict[str, Any] = {}

    # CPU
    metrics["cpu_percent"] = psutil.cpu_percent(interval=0)
    metrics["cpu_cores_logical"] = psutil.cpu_count(logical=True)
    metrics["cpu_cores_physical"] = psutil.cpu_count(logical=False)

    # Memory
    vm = psutil.virtual_memory()
    metrics["ram_total_gb"] = round(vm.total / (1024**3), 2)
    metrics["ram_used_gb"] = round(vm.used / (1024**3), 2)
    metrics["ram_percent"] = vm.percent

    # GPU (NVML)
    gpu_list = []
    if _HAS_NVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                # Handle both old (bytes) and new (str) API
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode()

                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_list.append(
                    {
                        "index": i,
                        "name": gpu_name,
                        "memory_used_gb": round(mem.used / (1024**3), 2),
                        "memory_total_gb": round(mem.total / (1024**3), 2),
                        "memory_percent": round((mem.used / mem.total) * 100, 1) if mem.total else 0.0,
                        "utilization_percent": util.gpu,
                        "temperature_c": temp,
                    }
                )
        except Exception as e:
            gpu_list.append({"error": f"NVML error: {e}"})
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    metrics["gpus"] = gpu_list
    return metrics


def update_history(metrics: Dict[str, Any]):
    """Add current metrics to historical data for graphing."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    history_timestamps.append(timestamp)
    history_cpu.append(metrics.get("cpu_percent", 0))
    history_ram.append(metrics.get("ram_percent", 0))

    # GPU metrics (first GPU if available)
    gpus = metrics.get("gpus", [])
    if gpus and "utilization_percent" in gpus[0]:
        history_gpu_util.append(gpus[0]["utilization_percent"])
        history_gpu_mem.append(gpus[0]["memory_percent"])
    else:
        history_gpu_util.append(0)
        history_gpu_mem.append(0)


def get_plot_data():
    """Create DataFrames for plotting resource usage over time."""
    if not history_timestamps:
        # Return empty DataFrames with proper structure
        return pd.DataFrame(columns=["Time", "Metric", "Usage %"]), pd.DataFrame(columns=["Time", "Metric", "Usage %"])

    # Convert to long format for Gradio LinePlot
    times = list(history_timestamps)

    # CPU and RAM plot - long format
    cpu_ram_data = []
    for i, time in enumerate(times):
        cpu_ram_data.append({"Time": i, "Metric": "CPU", "Usage %": history_cpu[i]})
        cpu_ram_data.append({"Time": i, "Metric": "RAM", "Usage %": history_ram[i]})
    cpu_ram_df = pd.DataFrame(cpu_ram_data)

    # GPU plot - long format
    gpu_data = []
    for i, time in enumerate(times):
        gpu_data.append({"Time": i, "Metric": "Utilization", "Usage %": history_gpu_util[i]})
        gpu_data.append({"Time": i, "Metric": "Memory", "Usage %": history_gpu_mem[i]})
    gpu_df = pd.DataFrame(gpu_data)

    return cpu_ram_df, gpu_df


def run_pipeline(image_path, yolo_model, sam2_model, marker_size_cm):
    """Run the complete pipeline and return detailed results."""

    # Load original image
    original_img = Image.open(image_path)
    img_name = image_path.split("/")[-1]

    # Build detailed results text
    result_text = "=" * 60 + "\n"
    result_text += "PIPELINE EXECUTION RESULTS\n"
    result_text += "=" * 60 + "\n\n"

    # Step 1: Detection
    result_text += "STEP 1: OBJECT DETECTION (YOLO)\n"
    result_text += "-" * 60 + "\n"
    detections = pipeline.detect_objects(model_name=yolo_model, image_name=img_name)
    boxes = [d.model_dump() for d in detections]

    result_text += f"Model: {yolo_model}\n"
    result_text += f"Objects detected: {len(boxes)}\n\n"

    for i, box in enumerate(boxes, 1):
        result_text += f"  Object {i}:\n"
        result_text += f"    - Label: {box['label']}\n"
        result_text += f"    - Confidence: {box['confidence']:.2%}\n"
        result_text += f"    - BBox: ({box['x1']}, {box['y1']}) → ({box['x2']}, {box['y2']})\n\n"

    # Step 2: Segmentation
    result_text += "\nSTEP 2: SEGMENTATION (SAM2)\n"
    result_text += "-" * 60 + "\n"
    segmented = pipeline.segment_objects(model_name=sam2_model, image_name=img_name, boxes=boxes)
    masks = [s.model_dump() for s in segmented]

    result_text += f"Model: {sam2_model}\n"
    result_text += f"Masks generated: {len(masks)}\n\n"

    for i, mask in enumerate(masks, 1):
        result_text += f"  Mask {i}:\n"
        result_text += f"    - Label: {mask['label']}\n"
        result_text += f"    - Mask score: {mask.get('mask_score', 0):.2%}\n\n"

    # Step 3: Measurement
    result_text += "\nSTEP 3: MEASUREMENT (ArUco)\n"
    result_text += "-" * 60 + "\n"
    measured, markers_count, scale = pipeline.measure_objects(
        image_name=img_name,
        boxes=masks,
        marker_size_cm=marker_size_cm,
        marker_id=None,
    )
    measurements = [m.model_dump() for m in measured]

    result_text += f"ArUco markers detected: {markers_count}\n"
    result_text += f"Scale: {scale:.2f} px/cm\n"
    result_text += f"Marker size: {marker_size_cm} cm\n"
    result_text += f"Objects measured: {len(measurements)}\n\n"

    for i, meas in enumerate(measurements, 1):
        result_text += f"  Measurement {i} ({meas['label']}):\n"
        if meas.get("area_cm2"):
            result_text += f"    - Area: {meas['area_cm2']:.2f} cm²\n"
            result_text += f"    - Perimeter: {meas['perimeter_cm']:.2f} cm\n"
            result_text += f"    - Width: {meas['width_cm']:.2f} cm\n"
            result_text += f"    - Height: {meas['height_cm']:.2f} cm\n"
            result_text += f"    - Angle: {meas['angle']:.2f}°\n\n"
        else:
            result_text += "    - No measurements available (no ArUco markers detected)\n\n"

    result_text += "=" * 60 + "\n"
    result_text += "PIPELINE COMPLETED SUCCESSFULLY\n"
    result_text += "=" * 60 + "\n"

    # Create visualizations
    # 1. Original image with bounding boxes
    img_with_boxes = original_img.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for box in boxes:
        coords = [(box["x1"], box["y1"]), (box["x2"], box["y2"])]
        draw.rectangle(coords, outline="red", width=3)
        draw.text((box["x1"], box["y1"] - 10), f"{box['label']} {box['confidence']:.2f}", fill="red")

    # 2. Image with mask overlay (first object)
    img_with_mask = original_img.copy()
    if masks and masks[0].get("mask_base64"):
        mask_img = decode_mask(masks[0]["mask_base64"])
        img_with_mask = overlay_mask_on_image(original_img, mask_img)

    return result_text, img_with_boxes, img_with_mask


def gradio_ui():
    with gr.Blocks(title="Image Analysis Pipeline (Local)", theme=gr.themes.Base()) as demo:
        gr.Markdown("# 🥔 Image Analysis Pipeline - Local Gradio UI")
        gr.Markdown("Upload an image, select models, and run detection, segmentation, and measurement.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="📤 Upload Image")
                yolo_model = gr.Dropdown(YOLO_MODELS, value=YOLO_MODELS[0], label="🎯 YOLO Model")
                sam2_model = gr.Dropdown(SAM2_MODELS, value=SAM2_MODELS[0], label="✂️ SAM2 Model")
                marker_size = gr.Number(value=4.9, label="📏 ArUco Marker Size (cm)")
                run_btn = gr.Button("▶️ Run Pipeline", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📊 Pipeline Results"):
                        result_text = gr.Textbox(label="Detailed Output", lines=20, max_lines=40, show_copy_button=True)

                    with gr.Tab("🎯 Detection"):
                        img_with_boxes = gr.Image(label="Bounding Boxes", type="pil")

                    with gr.Tab("🎨 Segmentation"):
                        img_with_mask = gr.Image(label="Mask Overlay", type="pil")

                    with gr.Tab("🖥️ System Monitor"):
                        with gr.Row():
                            monitor_play_btn = gr.Button("⏸️ Pause", variant="secondary", size="sm")

                        gr.Markdown("### 📈 CPU & RAM")
                        cpu_ram_plot = gr.LinePlot(
                            x="Time",
                            y="Usage %",
                            color="Metric",
                            y_lim=[0, 100],
                            height=300,
                            width=None,
                            show_label=False,
                            container=True,
                            x_title=None,
                        )

                        gr.Markdown("### 🎮 GPU")
                        gpu_plot = gr.LinePlot(
                            x="Time",
                            y="Usage %",
                            color="Metric",
                            y_lim=[0, 100],
                            height=300,
                            width=None,
                            show_label=False,
                            container=True,
                            x_title=None,
                        )

                        gr.Markdown("### 📊 Current Metrics")
                        sys_cpu = gr.Slider(0, 100, value=0, label="CPU %", interactive=False)
                        sys_ram = gr.Slider(0, 100, value=0, label="RAM %", interactive=False)
                        sys_info = gr.JSON(label="Details (RAM/GPU)")

        # State to track if monitoring is active
        monitoring_active = gr.State(True)

        def on_run(image_path, yolo, sam2, marker):
            if not image_path:
                return "❌ No image uploaded.", None, None, 0, 0, {}, None, None
            # Run the pipeline
            result_text, img_boxes, img_mask = run_pipeline(image_path, yolo, sam2, marker)
            # Get current system metrics
            metrics = get_system_metrics()
            update_history(metrics)
            cpu = float(metrics.get("cpu_percent", 0))
            ram = float(metrics.get("ram_percent", 0))
            cpu_ram_df, gpu_df = get_plot_data()
            return result_text, img_boxes, img_mask, cpu, ram, metrics, cpu_ram_df, gpu_df

        run_btn.click(
            on_run,
            inputs=[image_input, yolo_model, sam2_model, marker_size],
            outputs=[result_text, img_with_boxes, img_with_mask, sys_cpu, sys_ram, sys_info, cpu_ram_plot, gpu_plot],
        )

        # Live system metrics updater
        def update_system_metrics():
            m = get_system_metrics()
            update_history(m)
            cpu = float(m.get("cpu_percent", 0))
            ram = float(m.get("ram_percent", 0))
            cpu_ram_df, gpu_df = get_plot_data()
            return cpu, ram, m, cpu_ram_df, gpu_df

        metrics_timer = gr.Timer(1.0, active=True)
        metrics_timer.tick(fn=update_system_metrics, outputs=[sys_cpu, sys_ram, sys_info, cpu_ram_plot, gpu_plot])

        # Toggle pause/play for monitoring
        def toggle_monitoring(is_active):
            new_state = not is_active
            button_text = "▶️ Play" if not new_state else "⏸️ Pause"
            return new_state, button_text, gr.Timer(active=new_state)

        monitor_play_btn.click(
            toggle_monitoring,
            inputs=[monitoring_active],
            outputs=[monitoring_active, monitor_play_btn, metrics_timer],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    gradio_ui()
