import base64
from collections import deque
from datetime import datetime
from typing import Any, Dict

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import psutil

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
    """Decode base64 mask to numpy array (BGRA format)."""
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    img_data = base64.b64decode(base64_str)
    # Decode image from bytes
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img


def overlay_mask_on_image(original_img, mask_img):
    """Overlay a semi-transparent mask on the original image.

    Args:
        original_img: numpy array in BGR format
        mask_img: numpy array in BGRA format

    Returns:
        numpy array in RGB format (for Gradio display)
    """
    # Convert BGR to BGRA if needed
    if original_img.shape[2] == 3:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)

    # Ensure mask is BGRA
    if mask_img.shape[2] == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2BGRA)

    # Resize mask to match original image if needed
    if mask_img.shape[:2] != original_img.shape[:2]:
        mask_img = cv2.resize(
            mask_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LANCZOS4
        )

    # Blend images using alpha channel
    # Extract alpha channel from mask
    alpha_mask = mask_img[:, :, 3] / 255.0
    alpha_mask = np.expand_dims(alpha_mask, axis=2)

    # Composite: result = mask * alpha + original * (1 - alpha)
    result = (mask_img[:, :, :3] * alpha_mask + original_img[:, :, :3] * (1 - alpha_mask)).astype(np.uint8)

    # Convert BGR to RGB for Gradio
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


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
    """Run the complete pipeline and return structured results.

    Returns:
        summary_md (str): Markdown summary text
        det_df (pd.DataFrame): Detection results table
        seg_df (pd.DataFrame): Segmentation results table
        meas_df (pd.DataFrame): Measurement results table
        img_with_boxes (np.ndarray RGB): Image with detection boxes
        img_with_mask (np.ndarray RGB): Image with combined mask overlay
    """

    # Load original image with OpenCV
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img_name = image_path.split("/")[-1]

    # Step 1: Detection
    detections = pipeline.detect_objects(model_name=yolo_model, image_name=img_name)
    boxes = [d.model_dump() for d in detections]

    # Step 2: Segmentation
    segmented = pipeline.segment_objects(model_name=sam2_model, image_name=img_name, boxes=boxes)
    masks = [s.model_dump() for s in segmented]

    # Step 3: Measurement
    measured, markers_count, scale = pipeline.measure_objects(
        image_name=img_name,
        boxes=masks,
        marker_size_cm=marker_size_cm,
        marker_id=None,
    )
    measurements = [m.model_dump() for m in measured]

    # Build tables (DataFrames)
    det_rows = [
        {
            "Label": b["label"],
            "Confidence": round(float(b.get("confidence", 0.0)), 3),
            "x1": int(b.get("x1", 0)),
            "y1": int(b.get("y1", 0)),
            "x2": int(b.get("x2", 0)),
            "y2": int(b.get("y2", 0)),
        }
        for b in boxes
    ]
    det_df = (
        pd.DataFrame(det_rows, columns=["Label", "Confidence", "x1", "y1", "x2", "y2"])
        if det_rows
        else pd.DataFrame(columns=["Label", "Confidence", "x1", "y1", "x2", "y2"])
    )

    seg_rows = [
        {
            "Label": m.get("label", ""),
            "Mask score": round(float(m.get("mask_score", 0.0)), 3),
        }
        for m in masks
    ]
    seg_df = (
        pd.DataFrame(seg_rows, columns=["Label", "Mask score"])
        if seg_rows
        else pd.DataFrame(columns=["Label", "Mask score"])
    )

    meas_cols = ["Label", "Area (cm^2)", "Perimeter (cm)", "Width (cm)", "Height (cm)", "Angle (deg)"]
    meas_rows = []
    for meas in measurements:
        meas_rows.append(
            {
                "Label": meas.get("label", ""),
                "Area (cm^2)": round(float(meas.get("area_cm2", 0.0)), 2) if meas.get("area_cm2") else None,
                "Perimeter (cm)": round(float(meas.get("perimeter_cm", 0.0)), 2) if meas.get("perimeter_cm") else None,
                "Width (cm)": round(float(meas.get("width_cm", 0.0)), 2) if meas.get("width_cm") else None,
                "Height (cm)": round(float(meas.get("height_cm", 0.0)), 2) if meas.get("height_cm") else None,
                "Angle (deg)": round(float(meas.get("angle", 0.0)), 2) if meas.get("angle") is not None else None,
            }
        )
    meas_df = pd.DataFrame(meas_rows, columns=meas_cols) if meas_rows else pd.DataFrame(columns=meas_cols)

    # Summary markdown
    scale_text = (
        "No ArUco markers detected — measurement skipped."
        if (markers_count == 0 or not scale)
        else f"{scale:.2f} px/cm"
    )
    summary_md = (
        f"### Pipeline Summary\n\n"
        f"- Image: `{img_name}`\n"
        f"- YOLO Model: `{yolo_model}` — Detections: **{len(boxes)}**\n"
        f"- SAM2 Model: `{sam2_model}` — Masks: **{len(masks)}**\n"
        f"- ArUco markers: **{markers_count}**\n"
        f"- Scale: **{scale_text}**\n"
        f"- Marker size: **{marker_size_cm} cm**\n"
    )

    # Create visualizations
    # 1. Original image with bounding boxes (convert to RGB for Gradio)
    img_with_boxes = original_img.copy()
    for box in boxes:
        # Draw rectangle
        pt1 = (int(box["x1"]), int(box["y1"]))
        pt2 = (int(box["x2"]), int(box["y2"]))
        cv2.rectangle(img_with_boxes, pt1, pt2, color=(0, 0, 255), thickness=3)  # Red in BGR

        # Draw text label
        label = f"{box['label']} {box['confidence']:.2f}"
        text_pos = (int(box["x1"]), int(box["y1"]) - 10)
        cv2.putText(img_with_boxes, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Convert BGR to RGB for Gradio display
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # 2. Image with mask overlay (all masks)
    img_with_mask = original_img.copy()
    if masks:
        # Convert to BGRA for alpha compositing
        if img_with_mask.shape[2] == 3:
            img_with_mask = cv2.cvtColor(img_with_mask, cv2.COLOR_BGR2BGRA)

        # Create overlay accumulator
        height, width = img_with_mask.shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)

        for mask in masks:
            if mask.get("mask_base64"):
                try:
                    mask_img = decode_mask(mask["mask_base64"])  # numpy array BGRA

                    # Ensure mask is BGRA
                    if mask_img.shape[2] == 3:
                        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2BGRA)

                    # Resize mask to match original image if needed
                    if mask_img.shape[:2] != (height, width):
                        mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_LANCZOS4)

                    # Composite this mask onto the overlay using alpha blending
                    alpha = mask_img[:, :, 3:4] / 255.0
                    overlay[:, :, :3] = (overlay[:, :, :3] * (1 - alpha) + mask_img[:, :, :3] * alpha).astype(np.uint8)
                    overlay[:, :, 3] = np.maximum(overlay[:, :, 3], mask_img[:, :, 3])

                except Exception as e:
                    print(f"Warning: Failed to decode/overlay mask: {e}")

        # Final composite of original image with combined overlay
        alpha = overlay[:, :, 3:4] / 255.0
        img_with_mask[:, :, :3] = (img_with_mask[:, :, :3] * (1 - alpha) + overlay[:, :, :3] * alpha).astype(np.uint8)

        # Convert to RGB for Gradio
        img_with_mask = cv2.cvtColor(img_with_mask, cv2.COLOR_BGRA2RGB)

    return summary_md, det_df, seg_df, meas_df, img_with_boxes, img_with_mask


def gradio_ui():
    # Validate that models are available
    if not YOLO_MODELS:
        print("⚠️  Warning: No YOLO models found in models/ directory. Add .pt files to continue.")
    if not SAM2_MODELS:
        print("⚠️  Warning: No SAM2 models configured. Check config.py SAM2_MODEL_CONFIGS.")

    with gr.Blocks(title="Image Analysis Pipeline (Local)", theme=gr.themes.Base()) as demo:
        gr.Markdown("# 🥔 Image Analysis Pipeline - Local Gradio UI")
        gr.Markdown("Upload an image, select models, and run detection, segmentation, and measurement.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="📤 Upload Image")
                yolo_model = gr.Dropdown(
                    YOLO_MODELS,
                    value=YOLO_MODELS[0] if YOLO_MODELS else None,
                    label="🎯 YOLO Model",
                    info=f"{len(YOLO_MODELS)} model(s) found",
                )
                sam2_model = gr.Dropdown(
                    SAM2_MODELS,
                    value=SAM2_MODELS[0] if SAM2_MODELS else None,
                    label="✂️ SAM2 Model",
                    info=f"{len(SAM2_MODELS)} model(s) available",
                )
                marker_size = gr.Number(value=4.9, label="📏 ArUco Marker Size (cm)")
                run_btn = gr.Button("▶️ Run Pipeline", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📊 Pipeline Results"):
                        summary_md = gr.Markdown()
                        with gr.Row():
                            det_df_comp = gr.Dataframe(interactive=False, wrap=True, label="Detections")
                            seg_df_comp = gr.Dataframe(interactive=False, wrap=True, label="Segmentations")
                        meas_df_comp = gr.Dataframe(interactive=False, wrap=True, label="Measurements")

                    with gr.Tab("🎯 Detection"):
                        img_with_boxes = gr.Image(label="Bounding Boxes", type="numpy")

                    with gr.Tab("🎨 Segmentation"):
                        img_with_mask = gr.Image(label="Mask Overlay", type="numpy")

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
                return (
                    "❌ No image uploaded.",
                    pd.DataFrame(),
                    pd.DataFrame(),
                    pd.DataFrame(),
                    None,
                    None,
                    0,
                    0,
                    {},
                    None,
                    None,
                )
            # Run the pipeline
            summary_md, det_df, seg_df, meas_df, img_boxes, img_mask = run_pipeline(image_path, yolo, sam2, marker)
            # Get current system metrics
            metrics = get_system_metrics()
            update_history(metrics)
            cpu = float(metrics.get("cpu_percent", 0))
            ram = float(metrics.get("ram_percent", 0))
            cpu_ram_df, gpu_df = get_plot_data()
            return summary_md, det_df, seg_df, meas_df, img_boxes, img_mask, cpu, ram, metrics, cpu_ram_df, gpu_df

        run_btn.click(
            on_run,
            inputs=[image_input, yolo_model, sam2_model, marker_size],
            outputs=[
                summary_md,
                det_df_comp,
                seg_df_comp,
                meas_df_comp,
                img_with_boxes,
                img_with_mask,
                sys_cpu,
                sys_ram,
                sys_info,
                cpu_ram_plot,
                gpu_plot,
            ],
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

    demo.launch(server_name="localhost", server_port=7860)


if __name__ == "__main__":
    gradio_ui()
