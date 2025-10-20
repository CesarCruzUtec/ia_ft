"""
SAM2-based image segmentation.
"""

import gc
from pathlib import Path
from typing import List, Optional

import numpy as np
from config import SAM2_CHECKPOINTS_DIR, SAM2_CONFIG_DIR, SAM2_MODEL_CONFIGS
from core import DeviceManager
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from modules.segmentation.mask_utils import mask_to_base64
from modules.segmentation.models import SegmentationBox


class SAM2Segmentor:
    """
    SAM2 (Segment Anything Model 2) image segmentor with model caching.
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        checkpoints_directory: Path = SAM2_CHECKPOINTS_DIR,
        config_directory: str = SAM2_CONFIG_DIR,
    ):
        """
        Initialize SAM2 segmentor.

        Args:
            device_manager: Shared device manager instance
            checkpoints_directory: Directory containing SAM2 checkpoint files
            config_directory: Directory containing SAM2 config files
        """
        self.device_manager = device_manager
        self.checkpoints_directory = Path(checkpoints_directory)
        self.config_directory = config_directory

        self._current_model_name: Optional[str] = None
        self._model: Optional[object] = None
        self._predictor: Optional[SAM2ImagePredictor] = None

    def load_model(self, model_name: str) -> SAM2ImagePredictor:
        """
        Load a SAM2 model. Uses cache if model is already loaded.

        Args:
            model_name: Name of the SAM2 model (e.g., "sam2.1_hiera_tiny")

        Returns:
            SAM2ImagePredictor instance

        Raises:
            FileNotFoundError: If model checkpoint is not found
            ValueError: If model name is not recognized
        """
        # Return cached predictor if already loaded
        if self._current_model_name == model_name and self._predictor is not None:
            print(f"✓ Using cached SAM2 model: {model_name}")
            return self._predictor

        # Validate model name
        if model_name not in SAM2_MODEL_CONFIGS:
            raise ValueError(f"Unknown SAM2 model: {model_name}. Available models: {list(SAM2_MODEL_CONFIGS.keys())}")

        # Clear previous model from memory
        if self._model is not None:
            print(f"⟳ Releasing previous SAM2 model: {self._current_model_name}")
            del self._model
            del self._predictor
            gc.collect()
            self.device_manager.clear_cache()

        # Load new model
        checkpoint_path = self.checkpoints_directory / f"{model_name}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

        config_filename = SAM2_MODEL_CONFIGS[model_name]
        config_path = f"{self.config_directory}/{config_filename}"

        print(f"⟳ Loading SAM2 model: {model_name}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Config: {config_path}")

        self._model = build_sam2(config_path, str(checkpoint_path), device=self.device_manager.device)
        self._predictor = SAM2ImagePredictor(self._model)
        self._current_model_name = model_name

        print("✓ SAM2 model loaded successfully")
        return self._predictor

    def segment_boxes(self, image: np.ndarray, boxes: List[dict], model_name: str) -> List[SegmentationBox]:
        """
        Generate segmentation masks for bounding boxes.

        Args:
            image: Input image as numpy array (RGB format)
            boxes: List of bounding box dictionaries with keys: x1, y1, x2, y2, label, confidence
            model_name: Name of the SAM2 model to use

        Returns:
            List of SegmentationBox objects with masks

        Raises:
            ValueError: If no boxes are provided or predictor is not initialized
        """
        if not boxes:
            raise ValueError("At least one bounding box is required for segmentation")

        # Load model (uses cache if already loaded)
        predictor = self.load_model(model_name)

        # Set image for prediction
        print("⟳ Processing image with SAM2...")
        predictor.set_image(image)
        print("✓ Image processed")

        # Prepare input boxes as numpy array
        input_boxes = np.array([[box["x1"], box["y1"], box["x2"], box["y2"]] for box in boxes])

        print(f"⟳ Predicting masks for {len(boxes)} boxes...")
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Create SegmentationBox objects with masks
        segmented_boxes = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            print(f"  Mask {i + 1}/{len(masks)}: score = {score:.4f}")

            # Convert mask to base64
            mask_base64_str = mask_to_base64(mask)

            # Create SegmentationBox with all data
            segmented_boxes.append(
                SegmentationBox(
                    label=boxes[i]["label"],
                    confidence=boxes[i]["confidence"],
                    x1=boxes[i]["x1"],
                    y1=boxes[i]["y1"],
                    x2=boxes[i]["x2"],
                    y2=boxes[i]["y2"],
                    mask_base64=mask_base64_str,
                    mask_score=float(score),
                )
            )

        print(f"✓ Generated {len(segmented_boxes)} masks")
        return segmented_boxes

    def get_current_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        return self._current_model_name

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is currently loaded."""
        return self._current_model_name == model_name and self._predictor is not None

    def clear_model(self):
        """Clear the currently loaded model from memory."""
        if self._model is not None:
            print(f"⟳ Clearing SAM2 model: {self._current_model_name}")
            del self._model
            del self._predictor
            self._model = None
            self._predictor = None
            self._current_model_name = None
            gc.collect()
            self.device_manager.clear_cache()
