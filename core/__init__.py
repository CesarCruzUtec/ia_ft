"""
Core package containing shared components for the image analysis pipeline.
"""

from .device_manager import DeviceManager
from .image_manager import ImageManager

__all__ = ["DeviceManager", "ImageManager"]
