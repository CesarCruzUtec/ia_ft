# Manual Testing - Quick Reference

## 🚀 Three Ways to Test

### 1. Interactive Menu (Easiest)

```pwsh
python test_menu.py
```

Select from a menu which module to test.

### 2. Direct Script (Quick)

```pwsh
python manual_tests/test_detection.py
```

Run a specific test directly.

### 3. Web UI (Visual)

```pwsh
uv run python -m webui.app
```

Test the complete pipeline visually.

---

## 📁 Available Manual Tests

| Test Script              | What It Tests        | Input Needed          |
| ------------------------ | -------------------- | --------------------- |
| `test_device_manager.py` | CPU/GPU detection    | None (automatic)      |
| `inspect_model.py`       | YOLO model inspector | Model selection       |
| `test_image_manager.py`  | Image loading        | Image path            |
| `test_detection.py`      | YOLO detection       | Model + Image         |
| `test_segmentation.py`   | SAM2 segmentation    | Model + Image + Boxes |
| `test_measurement.py`    | ArUco measurement    | Image + Marker size   |
| `test_full_pipeline.py`  | Complete workflow    | All of the above      |

---

## 📝 Quick Examples

### Test Device Detection

```pwsh
python manual_tests/test_device_manager.py
```

No input needed. Shows CPU/GPU info.

### Test Image Loading

```pwsh
python manual_tests/test_image_manager.py
```

Input: `images/mayor/brotes/test.jpg`

### Test Object Detection

```pwsh
python manual_tests/test_detection.py
```

1. Select model: `1` (detection_model)
2. Enter image: `images/test.jpg`

### Test Full Pipeline

```pwsh
python manual_tests/test_full_pipeline.py
```

Follow prompts for model selection and image path.

---

## 💡 Tips

**Image Paths**

-   Filename only: `test.jpg` → Searches in `images/`
-   Relative: `mayor/brotes/test.jpg` → From `images/` directory
-   Absolute: `C:\full\path\to\image.jpg`

**Model Selection**

-   Scripts show available models automatically
-   Just enter the number

**Bounding Boxes** (for segmentation test)

-   Format: `x1,y1,x2,y2,label`
-   Example: `100,100,300,300,potato`
-   Press Enter on empty line when done

**ArUco Markers**

-   Image must contain visible ArUco markers
-   Default marker size: 4.9 cm
-   Dictionary used: DICT_4X4_50

---

## 🔍 Troubleshooting

**"No models found"**

```pwsh
# Make sure you have .pt files in models/ directory
ls models/*.pt
```

**"Image not found"**

```pwsh
# Use absolute path or ensure image is in images/ directory
python manual_tests/test_image_manager.py
# Then enter: C:\full\path\to\your\image.jpg
```

**"Import errors"**

```pwsh
# Install dependencies
uv pip install -e .
```

---

## 📚 Full Documentation

See `manual_tests/README.md` for detailed documentation.
