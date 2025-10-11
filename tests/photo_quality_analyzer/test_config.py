"""
Unit tests for configuration module.

Tests cover:
- Default configuration
- Custom configuration
- Validation
- Preset configurations
"""

import pytest
from src.photo_quality_analyzer.config import (
    QualityAnalyzerConfig,
    WeightsConfig,
    ThresholdsConfig,
    get_default_config,
    create_custom_config,
    get_conservative_config,
    get_permissive_config,
    get_sharpness_focused_config,
    get_exposure_focused_config
)


class TestWeightsConfig:
    """Tests for WeightsConfig."""

    def test_default_weights(self):
        """Default weights should be 60/40."""
        weights = WeightsConfig()
        assert weights.sharpness == 0.6
        assert weights.exposure == 0.4

    def test_weights_validate_success(self):
        """Valid weights should pass validation."""
        weights = WeightsConfig(sharpness=0.7, exposure=0.3)
        assert weights.validate()

    def test_weights_validate_failure(self):
        """Invalid weights should fail validation."""
        weights = WeightsConfig(sharpness=0.7, exposure=0.4)  # Sum = 1.1
        assert not weights.validate()

    def test_weights_to_dict(self):
        """Should convert to dictionary."""
        weights = WeightsConfig()
        weights_dict = weights.to_dict()

        assert 'sharpness' in weights_dict
        assert 'exposure' in weights_dict
        assert 'composition' in weights_dict


class TestThresholdsConfig:
    """Tests for ThresholdsConfig."""

    def test_default_thresholds(self):
        """Default thresholds should be correct."""
        thresholds = ThresholdsConfig()
        assert thresholds.high_quality_min == 70.0
        assert thresholds.acceptable_min == 50.0
        assert thresholds.low_quality_max == 49.0

    def test_sharpness_thresholds(self):
        """Sharpness thresholds should be correct."""
        thresholds = ThresholdsConfig()
        assert thresholds.sharpness_very_blurry == 30.0
        assert thresholds.sharpness_slightly_blurry == 50.0
        assert thresholds.sharpness_adequate == 70.0

    def test_exposure_thresholds(self):
        """Exposure thresholds should be correct."""
        thresholds = ThresholdsConfig()
        assert thresholds.exposure_severe == 30.0
        assert thresholds.exposure_poor == 50.0
        assert thresholds.exposure_acceptable == 70.0


class TestQualityAnalyzerConfig:
    """Tests for main QualityAnalyzerConfig."""

    def test_default_config_valid(self):
        """Default configuration should be valid."""
        config = QualityAnalyzerConfig()
        assert config.validate()

    def test_config_has_all_sections(self):
        """Config should have all required sections."""
        config = QualityAnalyzerConfig()

        assert hasattr(config, 'weights')
        assert hasattr(config, 'thresholds')
        assert hasattr(config, 'sharpness')
        assert hasattr(config, 'exposure')
        assert hasattr(config, 'performance')
        assert hasattr(config, 'edge_cases')
        assert hasattr(config, 'cache')
        assert hasattr(config, 'logging')

    def test_config_version(self):
        """Config should have algorithm version."""
        config = QualityAnalyzerConfig()
        assert config.algorithm_version == "v1.0"

    def test_config_to_dict(self):
        """Should convert config to dictionary."""
        config = QualityAnalyzerConfig()
        config_dict = config.to_dict()

        assert 'weights' in config_dict
        assert 'thresholds' in config_dict
        assert 'algorithm_version' in config_dict

    def test_invalid_weights_raises(self):
        """Invalid weights should raise ValueError."""
        config = QualityAnalyzerConfig()
        config.weights.sharpness = 0.8  # Now sums to > 1.0
        config.weights.exposure = 0.4

        with pytest.raises(ValueError, match="Invalid weights"):
            config.validate()

    def test_invalid_thresholds_raises(self):
        """Invalid thresholds should raise ValueError."""
        config = QualityAnalyzerConfig()
        config.thresholds.high_quality_min = 45.0  # Less than acceptable_min
        config.thresholds.acceptable_min = 50.0

        with pytest.raises(ValueError, match="high_quality_min must be"):
            config.validate()


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_valid_config(self):
        """Should return valid configuration."""
        config = get_default_config()
        assert config.validate()

    def test_is_independent_instance(self):
        """Each call should return independent instance."""
        config1 = get_default_config()
        config2 = get_default_config()

        # Modify one
        config1.weights.sharpness = 0.7

        # Other should be unchanged
        assert config2.weights.sharpness == 0.6


class TestCreateCustomConfig:
    """Tests for create_custom_config function."""

    def test_custom_weights(self):
        """Should create config with custom weights."""
        config = create_custom_config(
            weights={'sharpness': 0.7, 'exposure': 0.3}
        )

        assert config.weights.sharpness == 0.7
        assert config.weights.exposure == 0.3

    def test_custom_thresholds(self):
        """Should create config with custom thresholds."""
        config = create_custom_config(
            thresholds={'high_quality_min': 75.0}
        )

        assert config.thresholds.high_quality_min == 75.0
        # Other thresholds should remain default
        assert config.thresholds.acceptable_min == 50.0

    def test_validates_custom_config(self):
        """Should validate custom config."""
        # This should raise because weights don't sum to 1.0
        with pytest.raises(ValueError):
            create_custom_config(
                weights={'sharpness': 0.8, 'exposure': 0.5}
            )

    def test_preserves_defaults(self):
        """Unspecified values should remain default."""
        config = create_custom_config(
            weights={'sharpness': 0.7, 'exposure': 0.3}
        )

        # Thresholds should be default
        assert config.thresholds.high_quality_min == 70.0


class TestPresetConfigurations:
    """Tests for preset configuration functions."""

    def test_conservative_config(self):
        """Conservative config should have higher thresholds."""
        config = get_conservative_config()

        assert config.thresholds.high_quality_min == 80.0
        assert config.thresholds.acceptable_min == 60.0
        assert config.thresholds.high_quality_min > 70.0  # Higher than default

    def test_permissive_config(self):
        """Permissive config should have lower thresholds."""
        config = get_permissive_config()

        assert config.thresholds.high_quality_min == 60.0
        assert config.thresholds.acceptable_min == 40.0
        assert config.thresholds.high_quality_min < 70.0  # Lower than default

    def test_sharpness_focused_config(self):
        """Sharpness-focused config should weight sharpness higher."""
        config = get_sharpness_focused_config()

        assert config.weights.sharpness == 0.8
        assert config.weights.exposure == 0.2
        assert config.weights.sharpness > 0.6  # Higher than default

    def test_exposure_focused_config(self):
        """Exposure-focused config should weight exposure higher."""
        config = get_exposure_focused_config()

        assert config.weights.sharpness == 0.4
        assert config.weights.exposure == 0.6
        assert config.weights.exposure > 0.4  # Higher than default

    def test_all_presets_valid(self):
        """All preset configs should be valid."""
        presets = [
            get_conservative_config(),
            get_permissive_config(),
            get_sharpness_focused_config(),
            get_exposure_focused_config(),
        ]

        for config in presets:
            assert config.validate(), f"Preset config failed validation"


class TestPerformanceConfig:
    """Tests for PerformanceConfig defaults."""

    def test_performance_defaults(self):
        """Performance config should have sensible defaults."""
        config = get_default_config()

        assert config.performance.max_image_size == 1024
        assert config.performance.default_batch_size == 500
        assert config.performance.default_num_workers == 4
        assert config.performance.per_photo_timeout == 30


class TestCacheConfig:
    """Tests for CacheConfig defaults."""

    def test_cache_defaults(self):
        """Cache config should have sensible defaults."""
        config = get_default_config()

        assert config.cache.enable_cache is True
        assert config.cache.invalidate_on_hash_change is True
        assert config.cache.cache_ttl_days == 365
