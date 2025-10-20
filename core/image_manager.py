"""
Image manager for loading, caching, and searching images.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import IMAGES_DIR


class ImageManager:
    """
    Manages image loading, caching, and file searching.
    Implements caching to avoid reloading the same image multiple times.
    """

    def __init__(self, images_directory: Path = IMAGES_DIR):
        """
        Initialize ImageManager.

        Args:
            images_directory: Root directory to search for images
        """
        self.images_directory = Path(images_directory)
        self._current_image_name: Optional[str] = None
        self._current_image: Optional[np.ndarray] = None

    def load_image(self, image_name: str, force_reload: bool = False) -> np.ndarray:
        """
        Load an image by name. Uses cache if image was previously loaded.

        Args:
            image_name: Name of the image file
            force_reload: Force reload even if cached

        Returns:
            Image as numpy array (RGB format)

        Raises:
            FileNotFoundError: If image is not found
        """
        # Return cached image if available
        if not force_reload and self._current_image_name == image_name and self._current_image is not None:
            print(f"✓ Using cached image: {image_name}")
            return self._current_image

        # Search for image file
        image_path = self._search_image_file(image_name)
        if image_path is None:
            raise FileNotFoundError(f"Image not found: {image_name}")

        print(f"✓ Loading image from: {image_path}")

        # Load and convert to RGB numpy array
        image_array = cv2.imread(str(image_path))
        print(f"✓ Image loaded - Shape: {image_array.shape}, Dtype: {image_array.dtype}")

        # Cache the loaded image
        self._current_image_name = image_name
        self._current_image = image_array

        return image_array

    def _search_image_file(self, image_name: str) -> Optional[Path]:
        """
        Search for an image file recursively in the images directory.

        Args:
            image_name: Name of the image file to search for

        Returns:
            Full path to the image file, or None if not found
        """
        # If user provided an absolute path, validate and return it directly
        candidate = Path(image_name)
        if candidate.is_absolute():
            if candidate.is_file():
                return candidate
            return None

        # If the image_name contains path parts (subdirectories), try resolving
        # it relative to the images directory first. This handles inputs like
        # 'mayor/brotes/img.jpg' or 'mayor\\brotes\\img.jpg'.
        if len(candidate.parts) > 1:
            rel_candidate = self.images_directory.joinpath(*candidate.parts)
            if rel_candidate.is_file():
                return rel_candidate

        # Fallback: search recursively by basename to avoid passing absolute
        # or non-relative patterns to rglob on Windows.
        basename = candidate.name
        for path in self.images_directory.rglob(basename):
            if path.is_file():
                return path
        return None

    def get_current_image(self) -> Optional[np.ndarray]:
        """
        Get the currently cached image.

        Returns:
            Cached image array or None
        """
        return self._current_image

    def get_current_image_name(self) -> Optional[str]:
        """
        Get the name of the currently cached image.

        Returns:
            Cached image name or None
        """
        return self._current_image_name

    def clear_cache(self):
        """Clear the image cache."""
        self._current_image_name = None
        self._current_image = None
        print("✓ Image cache cleared")

    def is_image_cached(self, image_name: str) -> bool:
        """
        Check if an image is currently cached.

        Args:
            image_name: Name of the image to check

        Returns:
            True if the image is cached
        """
        return self._current_image_name == image_name and self._current_image is not None
