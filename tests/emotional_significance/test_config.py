"""
Tests for configuration module
"""

import pytest
from src.emotional_significance.config import (
    EmotionalConfig,
    get_default_config,
    create_custom_config,
    get_conservative_config,
    get_permissive_config,
    get_emotion_focused_config,
    get_intimacy_focused_config,
    FaceDetectionConfig,
    ScoringWeightsConfig
)


class TestEmotionalConfig:
    """Tests for EmotionalConfig."""

    def test_default_config_creation(self):
        """Test creation of default configuration."""
        config = get_default_config()

        assert config is not None
        assert config.face_detection is not None
        assert config.smile_detection is not None
        assert config.proximity is not None
        assert config.engagement is not None
        assert config.scoring_weights is not None
        assert config.tier_thresholds is not None

    def test_config_validation(self):
        """Test configuration validation."""
        config = get_default_config()
        assert config.validate() is True

    def test_invalid_weights_validation(self):
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError):
            create_custom_config(
                scoring_weights={
                    'face_presence_weight': 30.0,
                    'emotion_weight': 40.0,
                    'intimacy_weight': 20.0,
                    'engagement_weight': 20.0  # Total = 110 (invalid)
                }
            )

    def test_custom_config_creation(self):
        """Test creation of custom configuration."""
        config = create_custom_config(
            face_detection={'confidence_threshold': 0.7},
            tier_thresholds={'high_tier_min': 75.0}
        )

        assert config.face_detection.confidence_threshold == 0.7
        assert config.tier_thresholds.high_tier_min == 75.0

    def test_conservative_config(self):
        """Test conservative preset configuration."""
        config = get_conservative_config()

        assert config.face_detection.confidence_threshold > 0.5
        assert config.tier_thresholds.high_tier_min > 70.0

    def test_permissive_config(self):
        """Test permissive preset configuration."""
        config = get_permissive_config()

        assert config.face_detection.confidence_threshold < 0.5
        assert config.tier_thresholds.high_tier_min < 70.0

    def test_emotion_focused_config(self):
        """Test emotion-focused preset configuration."""
        config = get_emotion_focused_config()

        assert config.scoring_weights.emotion_weight > 40.0

    def test_intimacy_focused_config(self):
        """Test intimacy-focused preset configuration."""
        config = get_intimacy_focused_config()

        assert config.scoring_weights.intimacy_weight > 20.0

    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = get_default_config()
        config_dict = config.to_dict()

        assert 'face_detection' in config_dict
        assert 'scoring_weights' in config_dict
        assert 'tier_thresholds' in config_dict
        assert 'algorithm_version' in config_dict


class TestScoringWeightsConfig:
    """Tests for ScoringWeightsConfig."""

    def test_default_weights_sum_to_100(self):
        """Test that default weights sum to 100."""
        config = ScoringWeightsConfig()
        total = (config.face_presence_weight +
                 config.emotion_weight +
                 config.intimacy_weight +
                 config.engagement_weight)

        assert 99.0 <= total <= 101.0  # Allow floating point error

    def test_weights_validation(self):
        """Test weights validation."""
        config = ScoringWeightsConfig()
        assert config.validate() is True

    def test_weights_to_dict(self):
        """Test weights to dictionary conversion."""
        config = ScoringWeightsConfig()
        weights_dict = config.to_dict()

        assert 'face_presence' in weights_dict
        assert 'emotion' in weights_dict
        assert 'intimacy' in weights_dict
        assert 'engagement' in weights_dict


class TestFaceDetectionConfig:
    """Tests for FaceDetectionConfig."""

    def test_default_face_detection_config(self):
        """Test default face detection configuration."""
        config = FaceDetectionConfig()

        assert config.confidence_threshold == 0.5
        assert config.min_face_size_ratio == 0.05
        assert config.max_faces == 20
        assert config.dnn_input_size == (300, 300)
