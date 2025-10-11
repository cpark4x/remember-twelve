"""
Unit tests for main analyzer module.

Tests cover:
- Photo analysis from file path
- Image analysis from numpy array
- Batch processing
- Image loading and preprocessing
- Error handling
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

from src.photo_quality_analyzer.analyzer import (
    PhotoQualityAnalyzer,
    analyze_photo_simple
)
from src.photo_quality_analyzer.config import (
    create_custom_config,
    get_default_config
)
from src.photo_quality_analyzer.metrics.composite import QualityScore


class TestPhotoQualityAnalyzerInit:
    """Tests for PhotoQualityAnalyzer initialization."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        analyzer = PhotoQualityAnalyzer()
        assert analyzer.config is not None
        assert analyzer.config.weights.sharpness == 0.6

    def test_init_custom_config(self):
        """Should initialize with custom config."""
        custom_config = create_custom_config(
            weights={'sharpness': 0.7, 'exposure': 0.3}
        )
        analyzer = PhotoQualityAnalyzer(config=custom_config)
        assert analyzer.config.weights.sharpness == 0.7

    def test_init_validates_config(self):
        """Should validate config on initialization."""
        # This should work fine (valid config)
        valid_config = get_default_config()
        analyzer = PhotoQualityAnalyzer(config=valid_config)
        assert analyzer.config is not None


class TestAnalyzeImage:
    """Tests for analyze_image method."""

    def test_analyze_sharp_image(self):
        """Should analyze sharp image correctly."""
        analyzer = PhotoQualityAnalyzer()

        # Create sharp checkerboard pattern
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i:i+20, j:j+20] = 255

        score = analyzer.analyze_image(image)

        assert isinstance(score, QualityScore)
        assert score.sharpness > 50, "Sharp image should have high sharpness"
        assert 0 <= score.composite <= 100

    def test_analyze_blurry_image(self):
        """Should detect blurry image."""
        analyzer = PhotoQualityAnalyzer()

        # Create and blur image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i:i+20, j:j+20] = 255

        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        score = analyzer.analyze_image(blurred)

        assert score.sharpness < 50, "Blurred image should have low sharpness"

    def test_analyze_well_exposed_image(self):
        """Should detect well-exposed image."""
        analyzer = PhotoQualityAnalyzer()

        # Create well-exposed image (good mid-tone distribution)
        image = np.random.normal(128, 40, (200, 200, 3)).astype(np.uint8)
        score = analyzer.analyze_image(image)

        assert score.exposure > 40, "Well-exposed image should have decent exposure"

    def test_analyze_overexposed_image(self):
        """Should detect overexposed image."""
        analyzer = PhotoQualityAnalyzer()

        # Create overexposed image
        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        score = analyzer.analyze_image(image)

        assert score.exposure < 50, "Overexposed image should have low exposure score"

    def test_returns_quality_score_object(self):
        """Should return QualityScore object with all fields."""
        analyzer = PhotoQualityAnalyzer()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = analyzer.analyze_image(image)

        assert hasattr(score, 'sharpness')
        assert hasattr(score, 'exposure')
        assert hasattr(score, 'composite')
        assert hasattr(score, 'tier')

    def test_uses_custom_weights(self):
        """Should use custom weights from config."""
        # Create config with extreme sharpness weight
        config = create_custom_config(
            weights={'sharpness': 0.9, 'exposure': 0.1}
        )
        analyzer = PhotoQualityAnalyzer(config=config)

        # Create image with high sharpness, low exposure
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = analyzer.analyze_image(image)

        # Composite should be heavily influenced by sharpness
        expected_composite = score.sharpness * 0.9 + score.exposure * 0.1
        assert abs(score.composite - expected_composite) < 1


class TestAnalyzePhoto:
    """Tests for analyze_photo method (file-based)."""

    def test_analyze_photo_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        analyzer = PhotoQualityAnalyzer()

        with pytest.raises(FileNotFoundError):
            analyzer.analyze_photo('/nonexistent/path/photo.jpg')

    def test_analyze_photo_not_a_file(self):
        """Should raise ValueError for directory path."""
        analyzer = PhotoQualityAnalyzer()

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="not a file"):
                analyzer.analyze_photo(tmpdir)

    def test_analyze_photo_with_path_object(self):
        """Should accept Path objects."""
        analyzer = PhotoQualityAnalyzer()

        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            # Create simple image
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, image)

        try:
            # Test with Path object
            path_obj = Path(temp_path)
            score = analyzer.analyze_photo(path_obj)
            assert isinstance(score, QualityScore)
        finally:
            os.unlink(temp_path)

    def test_analyze_photo_from_file(self):
        """Should analyze photo from actual file."""
        analyzer = PhotoQualityAnalyzer()

        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            # Create checkerboard image
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            for i in range(0, 200, 20):
                for j in range(0, 200, 20):
                    if (i // 20 + j // 20) % 2 == 0:
                        image[i:i+20, j:j+20] = 255
            cv2.imwrite(temp_path, image)

        try:
            score = analyzer.analyze_photo(temp_path)
            assert isinstance(score, QualityScore)
            assert score.sharpness > 50  # Should be relatively sharp
        finally:
            os.unlink(temp_path)


class TestAnalyzeBatch:
    """Tests for analyze_batch method."""

    def test_batch_empty(self):
        """Empty batch should return empty list."""
        analyzer = PhotoQualityAnalyzer()
        scores = analyzer.analyze_batch([])
        assert len(scores) == 0

    def test_batch_single(self):
        """Single photo batch should work."""
        analyzer = PhotoQualityAnalyzer()

        # Create temporary image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, image)

        try:
            scores = analyzer.analyze_batch([temp_path])
            assert len(scores) == 1
            assert isinstance(scores[0], QualityScore)
        finally:
            os.unlink(temp_path)

    def test_batch_multiple(self):
        """Multiple photos should all be analyzed."""
        analyzer = PhotoQualityAnalyzer()
        temp_paths = []

        # Create multiple temporary images
        for _ in range(3):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
                temp_paths.append(temp_path)
                image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(temp_path, image)

        try:
            scores = analyzer.analyze_batch(temp_paths)
            assert len(scores) == 3
            assert all(isinstance(s, (QualityScore, type(None))) for s in scores)
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.unlink(path)

    def test_batch_handles_errors(self):
        """Batch should handle errors gracefully."""
        analyzer = PhotoQualityAnalyzer()

        # Mix valid and invalid paths
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            valid_path = f.name
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(valid_path, image)

        try:
            paths = [
                valid_path,
                '/nonexistent/file.jpg',  # Invalid
                valid_path,
            ]

            scores = analyzer.analyze_batch(paths)
            assert len(scores) == 3
            # First and third should succeed, middle should be None
            assert isinstance(scores[0], QualityScore)
            assert scores[1] is None
            assert isinstance(scores[2], QualityScore)
        finally:
            os.unlink(valid_path)


class TestImagePreprocessing:
    """Tests for _preprocess_image method."""

    def test_resize_large_image(self):
        """Large images should be resized."""
        analyzer = PhotoQualityAnalyzer()

        # Create large image
        large_image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        processed = analyzer._preprocess_image(large_image)

        # Should be resized to max 1024
        assert max(processed.shape[:2]) == 1024

    def test_preserve_small_image(self):
        """Small images should not be resized."""
        analyzer = PhotoQualityAnalyzer()

        # Create small image
        small_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        processed = analyzer._preprocess_image(small_image)

        # Should remain same size
        assert processed.shape == small_image.shape

    def test_maintain_aspect_ratio(self):
        """Resizing should maintain aspect ratio."""
        analyzer = PhotoQualityAnalyzer()

        # Create image with 2:1 aspect ratio
        image = np.random.randint(0, 255, (2000, 4000, 3), dtype=np.uint8)
        processed = analyzer._preprocess_image(image)

        # Should maintain 2:1 ratio
        height, width = processed.shape[:2]
        aspect_ratio = width / height
        assert abs(aspect_ratio - 2.0) < 0.1


class TestConfigManagement:
    """Tests for config get/update methods."""

    def test_get_config(self):
        """Should return current config."""
        analyzer = PhotoQualityAnalyzer()
        config = analyzer.get_config()

        assert config is not None
        assert config.weights.sharpness == 0.6

    def test_update_config(self):
        """Should update config."""
        analyzer = PhotoQualityAnalyzer()

        new_config = create_custom_config(
            weights={'sharpness': 0.7, 'exposure': 0.3}
        )
        analyzer.update_config(new_config)

        assert analyzer.config.weights.sharpness == 0.7

    def test_update_config_validates(self):
        """Should validate new config."""
        analyzer = PhotoQualityAnalyzer()

        # Create invalid config
        invalid_config = get_default_config()
        invalid_config.weights.sharpness = 0.8  # Will make sum > 1.0

        with pytest.raises(ValueError):
            analyzer.update_config(invalid_config)


class TestAnalyzePhotoSimple:
    """Tests for analyze_photo_simple convenience function."""

    def test_returns_dict(self):
        """Should return dictionary with scores."""
        # Create temporary image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, image)

        try:
            result = analyze_photo_simple(temp_path)

            assert isinstance(result, dict)
            assert 'sharpness' in result
            assert 'exposure' in result
            assert 'composite' in result
            assert 'tier' in result
        finally:
            os.unlink(temp_path)

    def test_uses_default_config(self):
        """Should use default configuration."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, image)

        try:
            result = analyze_photo_simple(temp_path)
            # Should use default weights (60/40)
            expected = result['sharpness'] * 0.6 + result['exposure'] * 0.4
            assert abs(result['composite'] - expected) < 0.1
        finally:
            os.unlink(temp_path)


class TestImageLoading:
    """Tests for _load_image method."""

    def test_load_jpeg(self):
        """Should load JPEG images."""
        analyzer = PhotoQualityAnalyzer()

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = Path(f.name)
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(temp_path), image)

        try:
            loaded = analyzer._load_image(temp_path)
            assert loaded is not None
            assert len(loaded.shape) == 3  # Should be color image
        finally:
            temp_path.unlink()

    def test_load_png(self):
        """Should load PNG images."""
        analyzer = PhotoQualityAnalyzer()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = Path(f.name)
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(temp_path), image)

        try:
            loaded = analyzer._load_image(temp_path)
            assert loaded is not None
        finally:
            temp_path.unlink()

    def test_load_invalid_raises(self):
        """Should raise ValueError for invalid image."""
        analyzer = PhotoQualityAnalyzer()

        # Create non-image file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"Not an image")

        try:
            with pytest.raises(ValueError):
                analyzer._load_image(temp_path)
        finally:
            temp_path.unlink()
