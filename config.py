"""
Configuration and constants for the image analysis pipeline.
"""

from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR  # Now config.py is at root, so BASE_DIR is PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
SAM2_CHECKPOINTS_DIR = PROJECT_ROOT / "meta-sam2" / "checkpoints"
SAM2_CONFIG_DIR = "configs/sam2.1"

# SAM2 model configurations
SAM2_MODEL_CONFIGS = {
    "sam2.1_hiera_tiny": "sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "sam2.1_hiera_l.yaml",
}

# ArUco marker settings
ARUCO_DICTIONARY = "DICT_4X4_50"  # Default ArUco dictionary type
DEFAULT_MARKER_SIZE_CM = 4.9  # Default marker size in centimeters

# Mask visualization settings
MASK_COLOR_RGB = (255, 30, 30)  # Red color for mask overlay
MASK_ALPHA = 0.6  # Transparency for mask overlay
MASK_BORDER_COLOR = (255, 255, 255, 204)  # White border color with alpha
MASK_BORDER_THICKNESS = 2

# Image processing settings
MASK_THRESHOLD = 127  # Binary threshold for mask conversion
