"""
Photo Sources Module - Unified interface for local and cloud photo sources.

This module provides:
- PhotoSource: Abstract base class for all photo sources
- LocalPhotoSource: Access photos from local file system
- GooglePhotosSource: Access photos from Google Photos (OAuth)
- PhotoSourceFactory: Create appropriate source based on config

Version: 1.0.0
"""

from .base import PhotoSource
from .factory import PhotoSourceFactory

__version__ = "1.0.0"

__all__ = [
    "PhotoSource",
    "PhotoSourceFactory",
]
