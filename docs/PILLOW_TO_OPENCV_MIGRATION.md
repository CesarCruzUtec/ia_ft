# Pillow to OpenCV Migration

## Summary
Successfully migrated all Pillow (PIL) implementations to OpenCV to resolve compatibility issues.

## Files Modified

### 1. `webui/app.py`
**Changes:**
- Removed `from PIL import Image, ImageDraw`
- Added `import cv2` and `import numpy as np`
- Updated `decode_mask()`: Now uses `cv2.imdecode()` instead of `Image.open()`
- Updated `overlay_mask_on_image()`: Replaced PIL alpha compositing with OpenCV alpha blending using numpy operations
- Updated `run_pipeline()`:
  - Changed `Image.open()` to `cv2.imread()`
  - Replaced `ImageDraw.Draw()` with `cv2.rectangle()` and `cv2.putText()` for bounding boxes
  - Converted PIL-based mask overlay to OpenCV alpha blending operations
  - Added proper BGR to RGB conversion for Gradio display

**Key Technical Changes:**
- PIL `Image.alpha_composite()` → OpenCV alpha blending: `result = mask * alpha + original * (1 - alpha)`
- PIL `Image.resize()` → `cv2.resize()` with `cv2.INTER_LANCZOS4`
- PIL color mode conversions → `cv2.cvtColor()` with appropriate flags
- All images now handled as numpy arrays instead of PIL Image objects

### 2. `modules/segmentation/mask_utils.py`
**Changes:**
- Removed `from PIL import Image` and `import io`
- Updated `mask_to_base64()`:
  - Replaced PIL image encoding with `cv2.imencode()`
  - Added RGBA to BGRA conversion for OpenCV compatibility
- Updated `base64_to_mask()`:
  - Replaced PIL decoding with `cv2.imdecode()`
  - Added proper handling of grayscale/BGRA/BGR formats

## Benefits

### 1. **Performance**
- OpenCV is generally faster for image operations
- Native numpy array operations (no PIL <-> numpy conversions)
- Better memory efficiency with direct buffer operations

### 2. **Compatibility**
- Resolves Pillow-related issues on your system
- Single image processing library (OpenCV already used in other modules)
- Better integration with existing detection/segmentation pipelines

### 3. **Consistency**
- All image operations now use the same library
- Unified color space handling (BGR/RGB)
- Consistent array-based image representation

## Color Space Notes

### Important: BGR vs RGB
OpenCV uses **BGR** color order by default, while most other libraries (including Gradio) use **RGB**. 

**Conversions applied:**
- `cv2.imread()` loads as BGR → Convert to RGB for Gradio with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
- Drawing operations use BGR colors: Red = `(0, 0, 255)` in BGR
- Final output images are converted to RGB before returning to Gradio

## Testing Recommendations

1. **Test the web UI:**
   ```powershell
   uv run python -m webui.app
   ```
   - Upload an image
   - Verify bounding boxes display correctly (red color)
   - Verify mask overlays show with proper transparency
   - Check that colors look correct in the output

2. **Test manual scripts:**
   ```powershell
   uv run python -m manual_tests.test_detection
   uv run python -m manual_tests.test_segmentation
   uv run python -m manual_tests.test_full_pipeline
   ```

3. **Verify mask encoding/decoding:**
   - Segmentation masks should encode properly to base64
   - Mask overlays should composite correctly with transparency
   - No color distortion or artifacts

## Potential Issues to Watch

1. **Alpha Channel Handling:**
   - Verify transparent regions in masks display correctly
   - Check that multiple mask overlays composite properly

2. **Image Quality:**
   - Ensure no degradation when resizing masks
   - Verify LANCZOS interpolation produces smooth results

3. **Color Accuracy:**
   - Verify red bounding boxes appear red (not blue)
   - Check mask colors match configuration in `config.py`

## Rollback (if needed)

If issues occur, you can reinstall Pillow and revert these changes:
```powershell
uv pip install Pillow
```

Then use git to revert:
```powershell
git checkout HEAD -- webui/app.py modules/segmentation/mask_utils.py
```

## Migration Date
October 20, 2025

## Notes
- The `meta-sam2/` directory was not modified as it's a third-party library
- All runtime PIL references have been removed from project code
- Lint warnings about cognitive complexity are pre-existing and unrelated to this migration
