# Real-Time Resource Monitoring

## Overview
The WebUI now includes real-time resource monitoring with interactive graphs showing CPU, RAM, and GPU utilization over time.

## Features

### 📈 Live Graphs
Two clean, real-time graphs showing resource usage over the last 60 seconds:

**📈 CPU & RAM Graph:**
- **CPU line** (one color): Processor utilization percentage
  - Spikes when running YOLO detection
  - Shows processing intensity
- **RAM line** (another color): System memory usage percentage
  - Gradually increases as models load
  - Shows memory consumption patterns

**🎮 GPU Graph:**
- **Utilization line**: GPU compute usage percentage
  - Jumps high during SAM2 segmentation inference
  - Shows when the GPU is actively processing
- **Memory line**: GPU VRAM usage percentage
  - Increases when models are loaded to GPU
  - Shows GPU memory allocation

**Graph Features:**
- ✨ Clean, minimal design with no clutter
- ⏱️ X-axis: Simple time increments (no overwhelming timestamps)
- 📊 Y-axis: Fixed 0-100% scale for easy comparison
- 🎨 Color-coded metrics for instant recognition
- 📐 Full-width plots for maximum visibility
- 🔄 Auto-updates every second

### ⏯️ Pause/Play Control
- **Pause Button** (⏸️): Stops real-time monitoring updates
- **Play Button** (▶️): Resumes real-time monitoring
- Located at the top of the System Monitor tab
- Monitoring starts automatically when the WebUI launches

### 📊 Current Metrics Display
Below the graphs, you'll find:
- **CPU Slider**: Current CPU usage percentage
- **RAM Slider**: Current RAM usage percentage
- **JSON Details**: Detailed information including:
  - CPU cores (logical/physical)
  - RAM total/used (GB)
  - GPU information (name, memory, utilization, temperature)

## Data History
- Stores the **last 60 data points** (1 minute of history at 1-second intervals)
- Uses a rolling buffer (oldest data is removed when new data arrives)
- Data persists during the session but resets when the app restarts

## Technical Details

### Dependencies
- `pandas>=2.0.0`: For creating DataFrames for LinePlot components
- `psutil>=5.9.8`: For CPU and RAM monitoring
- `nvidia-ml-py>=12.0.0`: For GPU monitoring

### Components
- **Collections.deque**: Efficient fixed-size buffers for historical data
- **Gradio LinePlot**: Interactive time-series visualization
- **Gradio Timer**: 1-second interval for automatic updates
- **Gradio State**: Tracks pause/play monitoring state

### Functions
- `update_history(metrics)`: Adds current metrics to historical buffers
- `get_plot_data()`: Creates pandas DataFrames for plotting
- `toggle_monitoring(is_active)`: Handles pause/play state changes
- `update_system_metrics()`: Fetches current metrics and updates all displays

## Usage Tips

1. **Monitor During Pipeline Execution**: Watch how detection and segmentation tasks affect system resources
2. **Identify Bottlenecks**: See if CPU, RAM, or GPU is the limiting factor
3. **Pause When Not Needed**: Save system resources by pausing monitoring when not actively analyzing
4. **Compare Different Models**: Run the pipeline with different YOLO/SAM2 models and observe resource differences

## Graph Interpretation

### What You'll See When Running the Pipeline:

**Before Pipeline Execution:**
- CPU: Low baseline (1-5%)
- RAM: Steady baseline showing loaded models
- GPU Utilization: Near 0% (idle)
- GPU Memory: Stable (models loaded in VRAM)

**During YOLO Detection (Step 1):**
- CPU: Brief spike (preprocessing, postprocessing)
- GPU Utilization: Quick spike to 40-80% (inference)
- Duration: ~50-100ms (very quick)

**During SAM2 Segmentation (Step 2):**
- GPU Utilization: Sustained high usage (60-100%)
- GPU Memory: May increase slightly
- Duration: Longer than detection (more compute-intensive)

**During ArUco Measurement (Step 3):**
- CPU: Moderate increase (marker detection, calculations)
- GPU: Returns to idle
- RAM: Stable

**Common Patterns:**
- **Flat lines = System idle**: Just monitoring, no active processing
- **Sharp spikes = Quick operations**: Image preprocessing, YOLO detection
- **Sustained plateaus = Heavy processing**: SAM2 segmentation running
- **Gradual increase in RAM = Model loading**: First run loads models into memory

## Example Use Cases

1. **Performance Tuning**: Identify which pipeline stage consumes most resources
2. **Model Comparison**: Compare resource usage between different model sizes (tiny/small/base/large)
3. **Capacity Planning**: Determine if your hardware can handle batch processing
4. **Troubleshooting**: Detect memory leaks or unusual resource consumption patterns
