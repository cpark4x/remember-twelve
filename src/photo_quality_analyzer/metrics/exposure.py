"""
Exposure Analysis Module

This module implements exposure scoring using histogram distribution analysis.
It detects over/under-exposed images by analyzing pixel distribution.

Algorithm: Histogram Distribution Analysis
- Fast histogram computation (O(n))
- No ML required
- Works for both B&W and color photos
"""

from typing import Dict, Tuple
import numpy as np
import cv2


def calculate_exposure_score(image: np.ndarray) -> float:
    """
    Calculate exposure score using histogram analysis.

    The algorithm analyzes the distribution of pixel intensities:
    - Overexposure: Too many pixels at maximum brightness (clipped highlights)
    - Underexposure: Too many pixels at minimum brightness (crushed shadows)
    - Well-exposed: Bell curve distribution with most pixels in mid-tones

    Args:
        image: Input image as numpy array (RGB or grayscale)
               Shape: (height, width, 3) for RGB or (height, width) for grayscale

    Returns:
        float: Exposure score from 0-100
               - 0-30: Severely over/underexposed (lost detail)
               - 30-50: Poor exposure (recoverable)
               - 50-70: Acceptable exposure
               - 70-100: Well-exposed

    Raises:
        ValueError: If image is None or has invalid dimensions
        TypeError: If image is not a numpy array

    Examples:
        >>> import cv2
        >>> image = cv2.imread('photo.jpg')
        >>> score = calculate_exposure_score(image)
        >>> print(f"Exposure: {score:.1f}")
        Exposure: 82.3

        >>> # Handle overexposed image
        >>> overexposed = cv2.imread('washed_out.jpg')
        >>> score = calculate_exposure_score(overexposed)
        >>> if score < 50:
        ...     print("Warning: Poor exposure detected")
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

    # Convert to grayscale for consistent analysis
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 1:
            gray = image.squeeze()
        else:
            raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
    else:
        gray = image

    # Calculate histogram (256 bins for 8-bit image)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize to probabilities

    # Detect clipping (overexposure)
    # Top 2% of intensity range (pixels >= 250)
    highlights_clipped = hist[250:].sum()

    # Detect crushing (underexposure)
    # Bottom 2% of intensity range (pixels <= 5)
    shadows_crushed = hist[:5].sum()

    # Calculate distribution (well-exposed has bell curve)
    # Middle 60% of intensity range (50-200)
    mid_tones = hist[50:200].sum()

    # Scoring formula
    # Penalize clipping and crushing (each can reduce score by up to 100 points)
    # Reward good mid-tone distribution (can add up to 50 points)
    clipping_penalty = highlights_clipped * 100
    crushing_penalty = shadows_crushed * 100
    distribution_bonus = mid_tones * 50

    # Calculate final score
    score = 100.0 - clipping_penalty - crushing_penalty + distribution_bonus

    # Clamp to 0-100 range
    score = max(0.0, min(100.0, score))

    return float(score)


def get_exposure_tier(score: float) -> str:
    """
    Convert exposure score to quality tier.

    Args:
        score: Exposure score (0-100)

    Returns:
        str: Quality tier ('severe', 'poor', 'acceptable', 'well_exposed')

    Examples:
        >>> get_exposure_tier(25.0)
        'severe'
        >>> get_exposure_tier(85.0)
        'well_exposed'
    """
    if score < 30:
        return 'severe'
    elif score < 50:
        return 'poor'
    elif score < 70:
        return 'acceptable'
    else:
        return 'well_exposed'


def analyze_histogram(image: np.ndarray) -> Dict[str, float]:
    """
    Perform detailed histogram analysis.

    Args:
        image: Input image as numpy array

    Returns:
        dict: Dictionary containing:
            - highlights_clipped (float): Percentage of overexposed pixels
            - shadows_crushed (float): Percentage of underexposed pixels
            - mid_tones (float): Percentage of pixels in mid-tone range
            - mean_intensity (float): Average pixel intensity (0-255)
            - std_intensity (float): Standard deviation of intensity

    Examples:
        >>> analysis = analyze_histogram(image)
        >>> print(f"Clipped highlights: {analysis['highlights_clipped']:.1%}")
        Clipped highlights: 5.2%
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()

    return {
        'highlights_clipped': float(hist[250:].sum()),
        'shadows_crushed': float(hist[:5].sum()),
        'mid_tones': float(hist[50:200].sum()),
        'mean_intensity': float(gray.mean()),
        'std_intensity': float(gray.std())
    }


def calculate_exposure_with_metadata(image: np.ndarray) -> dict:
    """
    Calculate exposure score with additional metadata.

    Args:
        image: Input image as numpy array

    Returns:
        dict: Dictionary containing:
            - score (float): Exposure score (0-100)
            - tier (str): Quality tier
            - histogram_analysis (dict): Detailed histogram metrics
            - dimensions (tuple): Image dimensions (height, width)

    Examples:
        >>> result = calculate_exposure_with_metadata(image)
        >>> print(result)
        {
            'score': 82.3,
            'tier': 'well_exposed',
            'histogram_analysis': {
                'highlights_clipped': 0.02,
                'shadows_crushed': 0.01,
                'mid_tones': 0.78,
                'mean_intensity': 125.5,
                'std_intensity': 45.2
            },
            'dimensions': (1920, 1080)
        }
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate score
    score = calculate_exposure_score(image)

    # Get detailed analysis
    histogram_analysis = analyze_histogram(image)

    return {
        'score': float(score),
        'tier': get_exposure_tier(score),
        'histogram_analysis': histogram_analysis,
        'dimensions': gray.shape
    }


def detect_exposure_issues(image: np.ndarray,
                          clipping_threshold: float = 0.05,
                          crushing_threshold: float = 0.05) -> Tuple[bool, list]:
    """
    Detect specific exposure issues in an image.

    Args:
        image: Input image as numpy array
        clipping_threshold: Threshold for overexposure detection (default: 5%)
        crushing_threshold: Threshold for underexposure detection (default: 5%)

    Returns:
        tuple: (has_issues: bool, issues: list of str)
               issues can include: 'overexposed', 'underexposed', 'low_contrast'

    Examples:
        >>> has_issues, issues = detect_exposure_issues(image)
        >>> if has_issues:
        ...     print(f"Exposure issues detected: {', '.join(issues)}")
        Exposure issues detected: overexposed, low_contrast
    """
    analysis = analyze_histogram(image)
    issues = []

    # Check for overexposure
    if analysis['highlights_clipped'] > clipping_threshold:
        issues.append('overexposed')

    # Check for underexposure
    if analysis['shadows_crushed'] > crushing_threshold:
        issues.append('underexposed')

    # Check for low contrast (narrow distribution)
    if analysis['std_intensity'] < 30:
        issues.append('low_contrast')

    has_issues = len(issues) > 0

    return has_issues, issues
