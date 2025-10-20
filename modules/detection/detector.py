"""
YOLO-based object detector.
"""

import gc
from pathlib import Path
from typing import List, Optional

from config import MODELS_DIR
from core import DeviceManager
from ultralytics import YOLO

from modules.detection.models import DetectionBox


class YOLODetector:
    """
    YOLO object detector with model caching and GPU optimization.
    """

    def __init__(self, device_manager: DeviceManager, models_directory: Path = MODELS_DIR):
        """
        Initialize YOLO detector.

        Args:
            device_manager: Shared device manager instance
            models_directory: Directory containing YOLO model files
        """
        self.device_manager = device_manager
        self.models_directory = Path(models_directory)

        self._current_model_name: Optional[str] = None
        self._model: Optional[YOLO] = None

    def load_model(self, model_name: str) -> YOLO:
        """
        Load a YOLO model. Uses cache if model is already loaded.

        Args:
            model_name: Name of the model file (without .pt extension)

        Returns:
            Loaded YOLO model

        Raises:
            FileNotFoundError: If model file is not found
        """
        # Return cached model if already loaded
        if self._current_model_name == model_name and self._model is not None:
            print(f"✓ Using cached YOLO model: {model_name}")
            return self._model

        # Clear previous model from memory
        if self._model is not None:
            print(f"⟳ Releasing previous YOLO model: {self._current_model_name}")
            del self._model
            gc.collect()
            self.device_manager.clear_cache()

        # Load new model
        model_path = self.models_directory / f"{model_name}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        print(f"⟳ Loading YOLO model: {model_path}")
        self._model = YOLO(str(model_path))
        self._current_model_name = model_name
        print("✓ YOLO model loaded successfully")

        return self._model

    def detect_objects(self, image, model_name: str, confidence_threshold: float = 0.25) -> List[DetectionBox]:
        """
        Detect objects in an image using YOLO.

        Args:
            image: Image array (numpy format)
            model_name: Name of the YOLO model to use
            confidence_threshold: Minimum confidence score for detections

        Returns:
            List of DetectionBox objects
        """
        # Load model (uses cache if already loaded)
        model = self.load_model(model_name)

        print("⟳ Running YOLO inference...")
        results = model(image)

        # Parse results into DetectionBox objects
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]

                # Filter by confidence threshold
                if confidence >= confidence_threshold:
                    detections.append(
                        DetectionBox(
                            label=label,
                            confidence=confidence,
                            x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2),
                        )
                    )

        print(f"✓ Detected {len(detections)} objects")
        return detections

    def get_current_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        return self._current_model_name

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is currently loaded."""
        return self._current_model_name == model_name and self._model is not None

    def clear_model(self):
        """Clear the currently loaded model from memory."""
        if self._model is not None:
            print(f"⟳ Clearing YOLO model: {self._current_model_name}")
            del self._model
            self._model = None
            self._current_model_name = None
            gc.collect()
            self.device_manager.clear_cache()
