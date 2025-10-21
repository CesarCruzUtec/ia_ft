"""
Image manager for loading, caching, and searching images.
"""

from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

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
        self._current_image_source: Optional[str] = None
        self._current_image: Optional[np.ndarray] = None

    def load_image(self, image_source: str, force_reload: bool = False) -> np.ndarray:
        """
        Load an image by source. Uses cache if image was previously loaded.

        Args:
            image_source: Source of the image (file path or URL)
            force_reload: Force reload even if cached

        Returns:
            Image as numpy array (RGB format)

        Raises:
            FileNotFoundError: If image is not found
        """
        # Return cached image if available
        if not force_reload and self._current_image_source == image_source and self._current_image is not None:
            print(f"✓ Using cached image: {image_source}")
            return self._current_image

        # Check if image_source is a URL
        if image_source.startswith("http://") or image_source.startswith("https://"):
            # Load image from URL
            print(f"✓ Loading image from URL: {image_source}")
            response = requests.get(image_source)
            if response.status_code != 200:
                raise FileNotFoundError(f"Image not found at URL: {image_source}")

            image_data = BytesIO(response.content)
            image_array = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
            if image_array is None:
                raise ValueError(f"Failed to decode image from URL: {image_source}")

            print(f"✓ Image loaded from URL - Shape: {image_array.shape}, Dtype: {image_array.dtype}")

            # Cache the loaded image
            self._current_image_source = image_source
            self._current_image = image_array

            return image_array

        # Search for image file
        image_path = self._search_image_file(image_source)
        if image_path is None:
            raise FileNotFoundError(f"Image not found: {image_source}")

        print(f"✓ Loading image from: {image_path}")

        # Load and convert to RGB numpy array
        image_array = cv2.imread(str(image_path))
        print(f"✓ Image loaded - Shape: {image_array.shape}, Dtype: {image_array.dtype}")

        # Cache the loaded image
        self._current_image_source = image_source
        self._current_image = image_array

        return image_array

    def _search_image_file(self, image_source: str) -> Optional[Path]:
        """
        Search for an image file recursively in the images directory.

        Args:
            image_source: Source of the image (file path or URL)

        Returns:
            Full path to the image file, or None if not found
        """
        # If user provided an absolute path, validate and return it directly
        candidate = Path(image_source)
        if candidate.is_absolute():
            if candidate.is_file():
                return candidate.absolute()
            return None

        # If the image_source contains path parts (subdirectories), try resolving
        # it relative to the images directory first. This handles inputs like
        # 'mayor/brotes/img.jpg' or 'mayor\\brotes\\img.jpg'.
        if len(candidate.parts) > 1:
            rel_candidate = self.images_directory.joinpath(*candidate.parts)
            if rel_candidate.is_file():
                return rel_candidate.absolute()

        # Fallback: search recursively by basename to avoid passing absolute
        # or non-relative patterns to rglob on Windows.
        basename = candidate.name
        for path in self.images_directory.rglob(basename):
            if path.is_file():
                return path.absolute()
        return None

    def get_current_image(self) -> Optional[np.ndarray]:
        """
        Get the currently cached image.

        Returns:
            Cached image array or None
        """
        return self._current_image

    def get_current_image_source(self) -> Optional[str]:
        """
        Get the source of the currently cached image.

        Returns:
            Cached image source or None
        """
        return self._current_image_source

    def clear_cache(self):
        """Clear the image cache."""
        self._current_image_source = None
        self._current_image = None
        print("✓ Image cache cleared")

    def is_image_cached(self, image_source: str) -> bool:
        """
        Check if an image is currently cached.

        Args:
            image_source: Source of the image to check

        Returns:
            True if the image is cached
        """
        return self._current_image_source == image_source and self._current_image is not None
