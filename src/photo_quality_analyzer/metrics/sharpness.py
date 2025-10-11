"""
Sharpness Detection Module

This module implements sharpness scoring using the Laplacian variance method.
It detects blur in images by analyzing edge detection variance.

Algorithm: Laplacian Variance Method
- Fast, deterministic, no ML required
- Works on grayscale conversion
- Industry standard for blur detection
"""

from typing import Union
import numpy as np
import cv2


def calculate_sharpness_score(image: np.ndarray) -> float:
    """
    Calculate sharpness score using Laplacian variance.

    The Laplacian operator detects edges in an image. A sharp image has many
    well-defined edges, resulting in high variance. A blurry image has fewer
    or softer edges, resulting in low variance.

    Args:
        image: Input image as numpy array (RGB or grayscale)
               Shape: (height, width, 3) for RGB or (height, width) for grayscale

    Returns:
        float: Sharpness score from 0-100
               - 0-30: Very blurry (motion blur, out of focus)
               - 30-50: Slightly blurry (acceptable for action shots)
               - 50-70: Adequate sharpness
               - 70-100: Sharp/very sharp

    Raises:
        ValueError: If image is None or has invalid dimensions
        TypeError: If image is not a numpy array

    Examples:
        >>> import cv2
        >>> image = cv2.imread('photo.jpg')
        >>> score = calculate_sharpness_score(image)
        >>> print(f"Sharpness: {score:.1f}")
        Sharpness: 78.5

        >>> # Convert RGB to BGR for OpenCV
        >>> from PIL import Image
        >>> pil_image = Image.open('photo.jpg')
        >>> image = np.array(pil_image)
        >>> image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        >>> score = calculate_sharpness_score(image)
    """
    # Input validation
    if image is None:
        raise ValueError("Image cannot be None")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be numpy array, got {type(image)}")

    if image.size == 0:
        raise ValueError("Image cannot be empty")

    if len(image.shape) not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D array, got shape {image.shape}")

    # Convert to grayscale if needed (faster processing)
    if len(image.shape) == 3:
        # Check if already grayscale (3 identical channels)
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 1:
            gray = image.squeeze()
        else:
            raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
    else:
        gray = image

    # Apply Laplacian operator for edge detection
    # CV_64F for more precision in variance calculation
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate variance (higher = more edges = sharper)
    variance = laplacian.var()

    # Normalize to 0-100 scale
    # Based on empirical testing:
    # - variance < 100 = very blurry
    # - variance > 1000 = sharp
    # Linear mapping: score = variance / 10
    score = min(100.0, variance / 10.0)

    return float(score)


def get_sharpness_tier(score: float) -> str:
    """
    Convert sharpness score to quality tier.

    Args:
        score: Sharpness score (0-100)

    Returns:
        str: Quality tier ('very_blurry', 'slightly_blurry', 'adequate', 'sharp')

    Examples:
        >>> get_sharpness_tier(25.0)
        'very_blurry'
        >>> get_sharpness_tier(75.0)
        'sharp'
    """
    if score < 30:
        return 'very_blurry'
    elif score < 50:
        return 'slightly_blurry'
    elif score < 70:
        return 'adequate'
    else:
        return 'sharp'


def calculate_sharpness_with_metadata(image: np.ndarray) -> dict:
    """
    Calculate sharpness score with additional metadata.

    Args:
        image: Input image as numpy array

    Returns:
        dict: Dictionary containing:
            - score (float): Sharpness score (0-100)
            - tier (str): Quality tier
            - variance (float): Raw Laplacian variance
            - dimensions (tuple): Image dimensions (height, width)

    Examples:
        >>> result = calculate_sharpness_with_metadata(image)
        >>> print(result)
        {
            'score': 78.5,
            'tier': 'sharp',
            'variance': 785.3,
            'dimensions': (1920, 1080)
        }
    """
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    # Calculate score
    score = min(100.0, variance / 10.0)

    return {
        'score': float(score),
        'tier': get_sharpness_tier(score),
        'variance': float(variance),
        'dimensions': gray.shape
    }
