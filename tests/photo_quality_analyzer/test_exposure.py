"""
Unit tests for exposure analysis module.

Tests cover:
- Overexposure detection
- Underexposure detection
- Well-exposed images
- Edge cases (black, white, mid-gray images)
- Histogram analysis
- Input validation
"""

import pytest
import numpy as np
import cv2

from src.photo_quality_analyzer.metrics.exposure import (
    calculate_exposure_score,
    get_exposure_tier,
    calculate_exposure_with_metadata,
    analyze_histogram,
    detect_exposure_issues
)


class TestExposureScore:
    """Tests for calculate_exposure_score function."""

    def test_well_exposed_image(self):
        """Image with good mid-tone distribution should score high."""
        # Create image with bell curve distribution (most pixels in mid-tones)
        image = np.random.normal(128, 40, (200, 200)).astype(np.uint8)
        score = calculate_exposure_score(image)
        assert score > 60, f"Well-exposed image should score >60, got {score}"

    def test_overexposed_image(self):
        """Image with many clipped highlights should score low."""
        # Create mostly white image (overexposed)
        image = np.full((200, 200), 255, dtype=np.uint8)
        score = calculate_exposure_score(image)
        assert score < 30, f"Overexposed image should score <30, got {score}"

    def test_underexposed_image(self):
        """Image with crushed shadows should score low."""
        # Create mostly black image (underexposed)
        image = np.full((200, 200), 0, dtype=np.uint8)
        score = calculate_exposure_score(image)
        assert score < 30, f"Underexposed image should score <30, got {score}"

    def test_mid_gray_image(self):
        """Image with only mid-tones should score well."""
        # Create image with only mid-gray values (good distribution)
        image = np.full((200, 200), 128, dtype=np.uint8)
        score = calculate_exposure_score(image)
        assert score > 50, f"Mid-gray image should score >50, got {score}"

    def test_partial_clipping(self):
        """Image with some clipping should score lower than no clipping."""
        # Well-exposed image
        well_exposed = np.random.normal(128, 40, (200, 200)).astype(np.uint8)
        well_score = calculate_exposure_score(well_exposed)

        # Add some clipped highlights
        partially_clipped = well_exposed.copy()
        partially_clipped[0:50, 0:50] = 255  # Overexpose corner
        clipped_score = calculate_exposure_score(partially_clipped)

        assert clipped_score < well_score, \
            "Partial clipping should reduce score"

    def test_score_range(self):
        """Score should always be in 0-100 range."""
        test_images = [
            np.zeros((50, 50), dtype=np.uint8),  # Black
            np.full((50, 50), 255, dtype=np.uint8),  # White
            np.full((50, 50), 128, dtype=np.uint8),  # Mid-gray
            np.random.randint(0, 255, (50, 50), dtype=np.uint8),  # Random
        ]

        for image in test_images:
            score = calculate_exposure_score(image)
            assert 0 <= score <= 100, f"Score must be 0-100, got {score}"

    def test_rgb_and_grayscale(self):
        """Should handle both RGB and grayscale images."""
        # Create grayscale image
        gray = np.random.normal(128, 40, (100, 100)).astype(np.uint8)
        gray_score = calculate_exposure_score(gray)

        # Convert to RGB
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rgb_score = calculate_exposure_score(rgb)

        # Scores should be identical (same pixel values)
        assert abs(gray_score - rgb_score) < 1, \
            f"RGB and grayscale should give same scores: {rgb_score} vs {gray_score}"


class TestExposureTier:
    """Tests for get_exposure_tier function."""

    def test_severe_tier(self):
        """Scores 0-29 should be 'severe'."""
        assert get_exposure_tier(0) == 'severe'
        assert get_exposure_tier(15) == 'severe'
        assert get_exposure_tier(29) == 'severe'

    def test_poor_tier(self):
        """Scores 30-49 should be 'poor'."""
        assert get_exposure_tier(30) == 'poor'
        assert get_exposure_tier(40) == 'poor'
        assert get_exposure_tier(49) == 'poor'

    def test_acceptable_tier(self):
        """Scores 50-69 should be 'acceptable'."""
        assert get_exposure_tier(50) == 'acceptable'
        assert get_exposure_tier(60) == 'acceptable'
        assert get_exposure_tier(69) == 'acceptable'

    def test_well_exposed_tier(self):
        """Scores 70-100 should be 'well_exposed'."""
        assert get_exposure_tier(70) == 'well_exposed'
        assert get_exposure_tier(85) == 'well_exposed'
        assert get_exposure_tier(100) == 'well_exposed'


class TestHistogramAnalysis:
    """Tests for analyze_histogram function."""

    def test_returns_all_fields(self):
        """Should return dict with all expected fields."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        analysis = analyze_histogram(image)

        assert 'highlights_clipped' in analysis
        assert 'shadows_crushed' in analysis
        assert 'mid_tones' in analysis
        assert 'mean_intensity' in analysis
        assert 'std_intensity' in analysis

    def test_overexposed_detection(self):
        """Should detect high percentage of clipped highlights."""
        # Create overexposed image
        image = np.full((100, 100), 255, dtype=np.uint8)
        analysis = analyze_histogram(image)

        # Should have nearly 100% clipped highlights
        assert analysis['highlights_clipped'] > 0.9, \
            f"Should detect clipping, got {analysis['highlights_clipped']}"

    def test_underexposed_detection(self):
        """Should detect high percentage of crushed shadows."""
        # Create underexposed image
        image = np.full((100, 100), 0, dtype=np.uint8)
        analysis = analyze_histogram(image)

        # Should have nearly 100% crushed shadows
        assert analysis['shadows_crushed'] > 0.9, \
            f"Should detect crushing, got {analysis['shadows_crushed']}"

    def test_mid_tone_detection(self):
        """Should detect high percentage of mid-tones in well-exposed image."""
        # Create image with mid-tone values (50-200 range)
        image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        analysis = analyze_histogram(image)

        # Should have high mid-tone percentage
        assert analysis['mid_tones'] > 0.8, \
            f"Should detect mid-tones, got {analysis['mid_tones']}"

    def test_mean_intensity(self):
        """Should calculate correct mean intensity."""
        # Create image with known mean
        image = np.full((100, 100), 100, dtype=np.uint8)
        analysis = analyze_histogram(image)

        assert abs(analysis['mean_intensity'] - 100) < 1, \
            f"Mean should be ~100, got {analysis['mean_intensity']}"


class TestExposureIssueDetection:
    """Tests for detect_exposure_issues function."""

    def test_no_issues_well_exposed(self):
        """Well-exposed image should have no issues."""
        # Create well-exposed image
        image = np.random.normal(128, 40, (100, 100)).astype(np.uint8)
        has_issues, issues = detect_exposure_issues(image)

        assert not has_issues, "Well-exposed image should have no issues"
        assert len(issues) == 0, f"Should have no issues, got {issues}"

    def test_detects_overexposure(self):
        """Should detect overexposure issue."""
        # Create overexposed image
        image = np.full((100, 100), 255, dtype=np.uint8)
        has_issues, issues = detect_exposure_issues(image)

        assert has_issues, "Should detect issues"
        assert 'overexposed' in issues, f"Should detect overexposure, got {issues}"

    def test_detects_underexposure(self):
        """Should detect underexposure issue."""
        # Create underexposed image
        image = np.full((100, 100), 0, dtype=np.uint8)
        has_issues, issues = detect_exposure_issues(image)

        assert has_issues, "Should detect issues"
        assert 'underexposed' in issues, f"Should detect underexposure, got {issues}"

    def test_detects_low_contrast(self):
        """Should detect low contrast issue."""
        # Create low contrast image (narrow range)
        image = np.random.randint(120, 130, (100, 100), dtype=np.uint8)
        has_issues, issues = detect_exposure_issues(image)

        assert has_issues, "Should detect issues"
        assert 'low_contrast' in issues, f"Should detect low contrast, got {issues}"

    def test_custom_thresholds(self):
        """Should respect custom thresholds."""
        # Create image with small amount of clipping
        image = np.random.normal(200, 30, (100, 100)).astype(np.uint8)

        # Strict threshold (should detect issue)
        has_issues_strict, _ = detect_exposure_issues(
            image, clipping_threshold=0.01
        )

        # Lenient threshold (should not detect issue)
        has_issues_lenient, _ = detect_exposure_issues(
            image, clipping_threshold=0.5
        )

        # Results should differ based on threshold
        assert has_issues_strict != has_issues_lenient or not has_issues_strict, \
            "Thresholds should affect detection"


class TestExposureWithMetadata:
    """Tests for calculate_exposure_with_metadata function."""

    def test_returns_all_fields(self):
        """Should return dict with all expected fields."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = calculate_exposure_with_metadata(image)

        assert 'score' in result
        assert 'tier' in result
        assert 'histogram_analysis' in result
        assert 'dimensions' in result

    def test_dimensions_correct(self):
        """Should return correct image dimensions."""
        image = np.random.randint(0, 255, (150, 200), dtype=np.uint8)
        result = calculate_exposure_with_metadata(image)

        assert result['dimensions'] == (150, 200)


class TestExposureValidation:
    """Tests for input validation."""

    def test_none_image_raises(self):
        """None image should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_exposure_score(None)

    def test_empty_image_raises(self):
        """Empty image should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_exposure_score(np.array([]))

    def test_wrong_type_raises(self):
        """Non-numpy array should raise TypeError."""
        with pytest.raises(TypeError, match="must be numpy array"):
            calculate_exposure_score([1, 2, 3])

    def test_wrong_dimensions_raises(self):
        """1D or 4D arrays should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            calculate_exposure_score(np.array([1, 2, 3]))
