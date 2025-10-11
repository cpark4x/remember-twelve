"""
Unit tests for sharpness detection module.

Tests cover:
- Basic functionality
- Edge cases (black images, white images, single color)
- Synthetic blur generation
- Input validation
"""

import pytest
import numpy as np
import cv2

from src.photo_quality_analyzer.metrics.sharpness import (
    calculate_sharpness_score,
    get_sharpness_tier,
    calculate_sharpness_with_metadata
)


class TestSharpnessScore:
    """Tests for calculate_sharpness_score function."""

    def test_sharp_checkerboard(self):
        """Sharp synthetic image should score high."""
        # Create sharp checkerboard pattern (high frequency edges)
        image = np.zeros((200, 200), dtype=np.uint8)
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i:i+20, j:j+20] = 255

        score = calculate_sharpness_score(image)
        assert score > 70, f"Sharp checkerboard should score >70, got {score}"

    def test_blurred_image(self):
        """Blurred image should score low."""
        # Create sharp image then blur it
        image = np.zeros((200, 200), dtype=np.uint8)
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i:i+20, j:j+20] = 255

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        score = calculate_sharpness_score(blurred)

        assert score < 50, f"Blurred image should score <50, got {score}"

    def test_blur_reduces_score(self):
        """Blurring should reduce sharpness score."""
        # Create sharp image
        image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        sharp_score = calculate_sharpness_score(image)

        # Blur the image
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        blur_score = calculate_sharpness_score(blurred)

        assert blur_score < sharp_score, "Blur should reduce sharpness score"

    def test_black_image(self):
        """Completely black image (no edges) should score very low."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        score = calculate_sharpness_score(image)
        assert score < 10, f"Black image should score <10, got {score}"

    def test_white_image(self):
        """Completely white image (no edges) should score very low."""
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        score = calculate_sharpness_score(image)
        assert score < 10, f"White image should score <10, got {score}"

    def test_single_color_image(self):
        """Single color image (no edges) should score very low."""
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        score = calculate_sharpness_score(image)
        assert score < 10, f"Single color image should score <10, got {score}"

    def test_score_range(self):
        """Score should always be in 0-100 range."""
        # Test various images
        test_images = [
            np.zeros((50, 50), dtype=np.uint8),  # Black
            np.full((50, 50), 255, dtype=np.uint8),  # White
            np.random.randint(0, 255, (50, 50), dtype=np.uint8),  # Random
        ]

        for image in test_images:
            score = calculate_sharpness_score(image)
            assert 0 <= score <= 100, f"Score must be 0-100, got {score}"

    def test_rgb_and_grayscale(self):
        """Should handle both RGB and grayscale images."""
        # Create test pattern
        pattern = np.zeros((100, 100), dtype=np.uint8)
        pattern[40:60, 40:60] = 255

        # Grayscale
        gray_score = calculate_sharpness_score(pattern)

        # RGB (same pattern in all channels)
        rgb = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
        rgb_score = calculate_sharpness_score(rgb)

        # Scores should be very close (allow small floating point difference)
        assert abs(gray_score - rgb_score) < 1, \
            f"RGB and grayscale should give similar scores: {rgb_score} vs {gray_score}"


class TestSharpnessTier:
    """Tests for get_sharpness_tier function."""

    def test_very_blurry_tier(self):
        """Scores 0-29 should be 'very_blurry'."""
        assert get_sharpness_tier(0) == 'very_blurry'
        assert get_sharpness_tier(15) == 'very_blurry'
        assert get_sharpness_tier(29) == 'very_blurry'

    def test_slightly_blurry_tier(self):
        """Scores 30-49 should be 'slightly_blurry'."""
        assert get_sharpness_tier(30) == 'slightly_blurry'
        assert get_sharpness_tier(40) == 'slightly_blurry'
        assert get_sharpness_tier(49) == 'slightly_blurry'

    def test_adequate_tier(self):
        """Scores 50-69 should be 'adequate'."""
        assert get_sharpness_tier(50) == 'adequate'
        assert get_sharpness_tier(60) == 'adequate'
        assert get_sharpness_tier(69) == 'adequate'

    def test_sharp_tier(self):
        """Scores 70-100 should be 'sharp'."""
        assert get_sharpness_tier(70) == 'sharp'
        assert get_sharpness_tier(85) == 'sharp'
        assert get_sharpness_tier(100) == 'sharp'


class TestSharpnessWithMetadata:
    """Tests for calculate_sharpness_with_metadata function."""

    def test_returns_all_fields(self):
        """Should return dict with all expected fields."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = calculate_sharpness_with_metadata(image)

        assert 'score' in result
        assert 'tier' in result
        assert 'variance' in result
        assert 'dimensions' in result

    def test_dimensions_correct(self):
        """Should return correct image dimensions."""
        image = np.random.randint(0, 255, (150, 200), dtype=np.uint8)
        result = calculate_sharpness_with_metadata(image)

        assert result['dimensions'] == (150, 200)


class TestSharpnessValidation:
    """Tests for input validation."""

    def test_none_image_raises(self):
        """None image should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_sharpness_score(None)

    def test_empty_image_raises(self):
        """Empty image should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_sharpness_score(np.array([]))

    def test_wrong_type_raises(self):
        """Non-numpy array should raise TypeError."""
        with pytest.raises(TypeError, match="must be numpy array"):
            calculate_sharpness_score([1, 2, 3])

    def test_wrong_dimensions_raises(self):
        """1D or 4D arrays should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            calculate_sharpness_score(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="must be 2D or 3D"):
            calculate_sharpness_score(np.zeros((10, 10, 3, 3)))
