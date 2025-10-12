"""
Integration tests for EmotionalAnalyzer

These tests verify the entire system works end-to-end with real images.
"""

import pytest
from pathlib import Path
from src.emotional_significance import EmotionalAnalyzer, get_default_config


class TestEmotionalAnalyzerIntegration:
    """Integration tests for full analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return EmotionalAnalyzer()

    @pytest.fixture
    def fixtures_dir(self):
        """Get path to test fixtures."""
        return Path(__file__).parent / 'fixtures'

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.config is not None
        assert analyzer.face_detector is not None
        assert analyzer.smile_detector is not None
        assert analyzer.proximity_calculator is not None
        assert analyzer.engagement_detector is not None

    def test_analyze_single_face(self, analyzer, fixtures_dir):
        """Test analyzing image with single face."""
        photo_path = fixtures_dir / 'single_face.jpg'

        if not photo_path.exists():
            pytest.skip("Test fixture not available")

        score = analyzer.analyze_photo(str(photo_path))

        # Basic assertions
        assert score is not None
        assert 0.0 <= score.composite <= 100.0
        assert score.tier in ['high', 'medium', 'low']
        assert 'processing_time_ms' in score.metadata

    def test_analyze_couple(self, analyzer, fixtures_dir):
        """Test analyzing image with couple."""
        photo_path = fixtures_dir / 'couple.jpg'

        if not photo_path.exists():
            pytest.skip("Test fixture not available")

        score = analyzer.analyze_photo(str(photo_path))

        assert score is not None
        assert 0.0 <= score.composite <= 100.0

    def test_analyze_group(self, analyzer, fixtures_dir):
        """Test analyzing image with group."""
        photo_path = fixtures_dir / 'group.jpg'

        if not photo_path.exists():
            pytest.skip("Test fixture not available")

        score = analyzer.analyze_photo(str(photo_path))

        assert score is not None
        assert 0.0 <= score.composite <= 100.0

    def test_analyze_landscape(self, analyzer, fixtures_dir):
        """Test analyzing landscape (no faces)."""
        photo_path = fixtures_dir / 'landscape.jpg'

        if not photo_path.exists():
            pytest.skip("Test fixture not available")

        score = analyzer.analyze_photo(str(photo_path))

        # Landscape should have no faces
        assert score.face_count == 0
        assert score.composite < 30.0  # Low score for no faces
        assert score.tier == 'low'

    def test_analyze_batch(self, analyzer, fixtures_dir):
        """Test batch analysis."""
        photos = [
            fixtures_dir / 'single_face.jpg',
            fixtures_dir / 'couple.jpg',
            fixtures_dir / 'landscape.jpg'
        ]

        # Filter existing photos
        existing_photos = [str(p) for p in photos if p.exists()]

        if not existing_photos:
            pytest.skip("Test fixtures not available")

        scores = analyzer.analyze_batch(existing_photos)

        assert len(scores) == len(existing_photos)
        assert all(s is not None for s in scores)

    def test_file_not_found(self, analyzer):
        """Test handling of missing file."""
        with pytest.raises(FileNotFoundError):
            analyzer.analyze_photo('nonexistent_photo.jpg')

    def test_invalid_file(self, analyzer, tmp_path):
        """Test handling of invalid image file."""
        invalid_file = tmp_path / 'invalid.jpg'
        invalid_file.write_text('not an image')

        with pytest.raises(ValueError):
            analyzer.analyze_photo(str(invalid_file))

    def test_config_update(self, analyzer):
        """Test updating configuration."""
        from src.emotional_significance import create_custom_config

        new_config = create_custom_config(
            face_detection={'confidence_threshold': 0.7}
        )

        analyzer.update_config(new_config)

        assert analyzer.config.face_detection.confidence_threshold == 0.7

    def test_score_attributes(self, analyzer, fixtures_dir):
        """Test that score has all expected attributes."""
        photo_path = fixtures_dir / 'single_face.jpg'

        if not photo_path.exists():
            pytest.skip("Test fixture not available")

        score = analyzer.analyze_photo(str(photo_path))

        # Check all required attributes
        assert hasattr(score, 'face_count')
        assert hasattr(score, 'face_coverage')
        assert hasattr(score, 'emotion_score')
        assert hasattr(score, 'intimacy_score')
        assert hasattr(score, 'engagement_score')
        assert hasattr(score, 'composite')
        assert hasattr(score, 'tier')
        assert hasattr(score, 'metadata')

    def test_performance_target(self, analyzer, fixtures_dir):
        """Test that analysis meets performance target (<50ms)."""
        photo_path = fixtures_dir / 'single_face.jpg'

        if not photo_path.exists():
            pytest.skip("Test fixture not available")

        score = analyzer.analyze_photo(str(photo_path))

        processing_time = score.metadata.get('processing_time_ms', 0)

        # Allow generous margin for test environment
        # Target is <50ms, but allow up to 200ms for slower test systems
        assert processing_time < 200.0, f"Processing took {processing_time:.1f}ms (target <50ms)"

    def test_score_to_dict(self, analyzer, fixtures_dir):
        """Test score serialization."""
        photo_path = fixtures_dir / 'single_face.jpg'

        if not photo_path.exists():
            pytest.skip("Test fixture not available")

        score = analyzer.analyze_photo(str(photo_path))
        score_dict = score.to_dict()

        # Check dictionary contains expected keys
        assert 'face_count' in score_dict
        assert 'face_coverage' in score_dict
        assert 'emotion_score' in score_dict
        assert 'intimacy_score' in score_dict
        assert 'engagement_score' in score_dict
        assert 'composite' in score_dict
        assert 'tier' in score_dict
        assert 'metadata' in score_dict

    def test_analyze_from_numpy_array(self, analyzer, fixtures_dir):
        """Test analyzing from numpy array."""
        import cv2

        photo_path = fixtures_dir / 'single_face.jpg'

        if not photo_path.exists():
            pytest.skip("Test fixture not available")

        # Load image as numpy array
        image = cv2.imread(str(photo_path))

        score = analyzer.analyze_image(image)

        assert score is not None
        assert 0.0 <= score.composite <= 100.0
