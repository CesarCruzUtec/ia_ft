# Manual Testing Scripts

Simple interactive scripts to manually test each module with your own inputs.

## Quick Start

Just run any script and follow the prompts:

```pwsh
# Test device detection
python -m manual_tests.test_device_manager

# Test image loading
python -m manual_tests.test_image_manager

# Test object detection (YOLO)
python -m manual_tests.test_detection

# Test segmentation (SAM2)
python -m manual_tests.test_segmentation

# Test measurement (ArUco)
python -m manual_tests.test_measurement

# Test complete pipeline
python -m manual_tests.test_full_pipeline

# Inspect YOLO model files
python -m manual_tests.inspect_model
```

## What Each Script Does

### 1. test_device_manager.py
Tests device detection (CPU/GPU).

**No input needed** - just displays device info.

```pwsh
python -m manual_tests.test_device_manager
```

### 1b. inspect_model.py
Inspects YOLO model files to show configuration, architecture, and metadata.

**Input needed:**
- Model selection (or 'all' to compare)

```pwsh
python -m manual_tests.inspect_model

# Shows:
# - Model size, epochs, classes
# - Class names and IDs
# - Training configuration
# - Architecture details
# - Compare multiple models side-by-side
```

### 2. test_image_manager.py
Tests loading images from disk.

**Input needed:**
- Path to an image file

```pwsh
python manual_tests/test_image_manager.py

# You'll be prompted:
# Enter image path: images/mayor/brotes/test.jpg
```

### 3. test_detection.py
Tests YOLO object detection.

**Input needed:**
- YOLO model selection (from available models)
- Path to an image

```pwsh
python manual_tests/test_detection.py

# You'll be prompted:
# 1. Select model (1-5): 1
# 2. Enter image path: images/test.jpg
```

### 4. test_segmentation.py
Tests SAM2 segmentation.

**Input needed:**
- SAM2 model selection
- Path to an image
- Bounding boxes (format: x1,y1,x2,y2,label)

```pwsh
python manual_tests/test_segmentation.py

# You'll be prompted:
# 1. Select model (1-4): 1
# 2. Enter image path: images/test.jpg
# 3. Box 1: 100,100,200,200,potato
# 4. Box 2: (press Enter to finish)
```

### 5. test_measurement.py
Tests ArUco marker detection and measurement.

**Input needed:**
- Path to an image with ArUco markers
- Marker size in cm

```pwsh
python manual_tests/test_measurement.py

# You'll be prompted:
# 1. Enter image path: images/test.jpg
# 2. Enter marker size in cm (default: 4.9): 4.9
```

### 6. test_full_pipeline.py
Tests the complete workflow: Detection → Segmentation → Measurement

**Input needed:**
- YOLO model selection
- SAM2 model selection
- Path to an image
- ArUco marker size

```pwsh
python manual_tests/test_full_pipeline.py

# You'll be prompted for all inputs step-by-step
```

## Example Workflow

### Test Detection on Your Image

```pwsh
python manual_tests/test_detection.py
```

```
🎯 Manual Test: YOLO Detection
====================

1. Initializing components...
✓ Device: cpu

2. Available YOLO models:
   1. detection_model
   2. deteccion_papa_brotes_yolov11m_100epocas
   3. deteccion_papa_heridas_yolov11m_100epocas

Select model (1-3): 2

3. Select image to analyze
   You can provide:
     - Filename (e.g., 'test.jpg')
     - Relative path (e.g., 'mayor/brotes/image.jpg')
     - Absolute path

Enter image path: images/mayor/brotes/PAPA AMARILLA PROC - BROTES 3.jpeg

4. Loading image...
✓ Image loaded: (1600, 1200, 3)

5. Running YOLO detection...
✓ Detected 5 object(s)

Object 1:
  - Label: brotes
  - Confidence: 95.32%
  - Bounding Box: (150, 200) → (250, 350)
  ...
```

### Test Full Pipeline

```pwsh
python manual_tests/test_full_pipeline.py
```

Follow the prompts to:
1. Select YOLO model
2. Select SAM2 model
3. Provide image path
4. Enter marker size

The script will run all three steps and show results.

## Tips

1. **Image Paths**: You can use:
   - Just filename: `test.jpg` (searches in images/)
   - Relative: `mayor/brotes/test.jpg`
   - Absolute: `C:\path\to\image.jpg`

2. **Model Selection**: Scripts show available models automatically

3. **Bounding Boxes** (for segmentation):
   - Format: `x1,y1,x2,y2,label`
   - Example: `100,100,200,200,potato`
   - Press Enter on empty line to finish

4. **ArUco Markers**: Make sure your test image contains visible ArUco markers for measurement tests

## Common Use Cases

### Quick Check After Code Changes
```pwsh
# Test the module you modified
python manual_tests/test_detection.py
```

### Test New Model
```pwsh
# Add your model to models/ directory
# Then run
python manual_tests/test_detection.py
# Select your new model from the list
```

### Debug Detection Issues
```pwsh
# Run detection test to see exact detections
python manual_tests/test_detection.py
# Check confidence scores and bounding boxes
```

### Verify Measurements
```pwsh
# Test with image containing ArUco markers
python manual_tests/test_measurement.py
# Verify scale calculation and measurements
```

## Differences from Automated Tests

| Manual Tests | Automated Tests (pytest) |
|-------------|--------------------------|
| ✅ Interactive prompts | ❌ No interaction |
| ✅ Use your own images | ❌ Use sample data |
| ✅ Select models manually | ❌ Use mocks |
| ✅ See detailed output | ❌ Pass/fail only |
| ✅ Good for debugging | ❌ Good for CI/CD |
| ✅ Test real scenarios | ❌ Test edge cases |

Use **manual tests** when you want to:
- Test with real images
- Try different models
- Debug issues
- Verify outputs visually

Use **automated tests** (pytest) when you want to:
- Run tests in CI/CD
- Verify code changes don't break things
- Test edge cases
- Get coverage reports
