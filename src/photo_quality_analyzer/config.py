"""
Photo Quality Analyzer Configuration

This module contains all configurable parameters for the quality analysis system.
Values are tuned based on empirical testing and can be adjusted without code changes.

Configuration includes:
- Score weights for composite calculation
- Quality tier thresholds
- Sharpness detection parameters
- Exposure detection parameters
- Performance settings
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class WeightsConfig:
    """Configuration for score weights in composite calculation."""

    # Composite score weights (must sum to 1.0)
    sharpness: float = 0.6  # 60% weight - users care most about blur
    exposure: float = 0.4   # 40% weight - exposure issues are more forgiving

    # Reserved for future enhancements
    composition: float = 0.0  # Not implemented in Phase 1

    def validate(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = self.sharpness + self.exposure + self.composition
        return 0.99 <= total <= 1.01  # Allow small floating point error

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for use in calculations."""
        return {
            'sharpness': self.sharpness,
            'exposure': self.exposure,
            'composition': self.composition
        }


@dataclass
class ThresholdsConfig:
    """Configuration for quality tier thresholds."""

    # Composite score thresholds
    high_quality_min: float = 70.0      # Score >= 70: Prioritize for curation
    acceptable_min: float = 50.0        # Score >= 50: Include if needed
    low_quality_max: float = 49.0       # Score < 50: Exclude from curation

    # Sharpness thresholds (Laplacian variance)
    sharpness_very_blurry: float = 30.0      # Motion blur, out of focus
    sharpness_slightly_blurry: float = 50.0  # Acceptable for action shots
    sharpness_adequate: float = 70.0         # Good enough quality
    # Above 70.0 = sharp/very sharp

    # Exposure thresholds
    exposure_severe: float = 30.0       # Severely over/underexposed
    exposure_poor: float = 50.0         # Poor but recoverable
    exposure_acceptable: float = 70.0   # Acceptable quality
    # Above 70.0 = well-exposed


@dataclass
class SharpnessConfig:
    """Configuration for sharpness detection algorithm."""

    # Laplacian variance thresholds (empirically determined)
    variance_min: float = 100.0    # Variance < 100 = very blurry
    variance_max: float = 1000.0   # Variance > 1000 = sharp

    # Normalization factor (converts variance to 0-100 scale)
    # score = min(100, variance / normalization_factor)
    normalization_factor: float = 10.0

    # Edge case handling
    handle_motion_blur: bool = True     # Accept motion blur (don't distinguish)
    handle_soft_focus: bool = False     # No special handling in Phase 1


@dataclass
class ExposureConfig:
    """Configuration for exposure detection algorithm."""

    # Histogram thresholds
    highlights_threshold: int = 250     # Pixel values >= 250 = clipped
    shadows_threshold: int = 5          # Pixel values <= 5 = crushed

    # Mid-tone range (for distribution analysis)
    mid_tone_min: int = 50              # Lower bound of mid-tone range
    mid_tone_max: int = 200             # Upper bound of mid-tone range

    # Penalty and bonus multipliers
    clipping_penalty_multiplier: float = 100.0      # How much to penalize clipping
    crushing_penalty_multiplier: float = 100.0      # How much to penalize crushing
    distribution_bonus_multiplier: float = 50.0     # Reward for good distribution

    # Issue detection thresholds (as percentage)
    clipping_issue_threshold: float = 0.05  # 5% clipped = overexposure issue
    crushing_issue_threshold: float = 0.05  # 5% crushed = underexposure issue
    low_contrast_threshold: float = 30.0    # Std dev < 30 = low contrast


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    # Image preprocessing
    max_image_size: int = 1024          # Resize to max 1024px (maintains quality)
    thumbnail_quality: int = 95         # JPEG quality for resizing (1-100)

    # Batch processing
    default_batch_size: int = 500       # Photos per batch
    default_num_workers: int = 4        # Parallel workers

    # Timeouts and limits
    per_photo_timeout: int = 30         # Seconds per photo analysis
    max_retries: int = 3                # Retry attempts on failure

    # Memory management
    enable_memory_cleanup: bool = True  # Clean up between batches
    max_memory_mb: int = 2048          # Target max memory usage (2GB)


@dataclass
class EdgeCaseConfig:
    """Configuration for edge case handling (future enhancements)."""

    # Night photo detection (Phase 2)
    handle_night_photos: bool = False
    night_photo_threshold: float = 50.0  # Mean intensity < 50 = night photo

    # Screenshot detection (Phase 2)
    detect_screenshots: bool = False
    screenshot_aspect_ratios: list = field(default_factory=lambda: [16/9, 16/10, 4/3])

    # Text-heavy image detection (Phase 2)
    detect_text_images: bool = False


@dataclass
class CacheConfig:
    """Configuration for score caching."""

    enable_cache: bool = True
    invalidate_on_hash_change: bool = True
    cache_ttl_days: int = 365  # Cache expires after 1 year


@dataclass
class LoggingConfig:
    """Configuration for logging and debugging."""

    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_analysis_time: bool = True
    log_score_distribution: bool = True
    log_errors: bool = True


@dataclass
class QualityAnalyzerConfig:
    """
    Master configuration for Photo Quality Analyzer.

    Contains all sub-configurations for different components.
    """

    weights: WeightsConfig = field(default_factory=WeightsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    sharpness: SharpnessConfig = field(default_factory=SharpnessConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    edge_cases: EdgeCaseConfig = field(default_factory=EdgeCaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Algorithm version (for future upgrades)
    algorithm_version: str = "v1.0"

    def validate(self) -> bool:
        """Validate entire configuration."""
        if not self.weights.validate():
            raise ValueError("Invalid weights: must sum to 1.0")

        if self.thresholds.high_quality_min <= self.thresholds.acceptable_min:
            raise ValueError("high_quality_min must be > acceptable_min")

        if self.thresholds.acceptable_min <= self.thresholds.low_quality_max:
            raise ValueError("acceptable_min must be > low_quality_max")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary."""
        return {
            'weights': {
                'sharpness': self.weights.sharpness,
                'exposure': self.weights.exposure,
                'composition': self.weights.composition
            },
            'thresholds': {
                'high_quality_min': self.thresholds.high_quality_min,
                'acceptable_min': self.thresholds.acceptable_min,
                'low_quality_max': self.thresholds.low_quality_max
            },
            'algorithm_version': self.algorithm_version
        }


# Default configuration instance
DEFAULT_CONFIG = QualityAnalyzerConfig()


def get_default_config() -> QualityAnalyzerConfig:
    """
    Get default configuration.

    Returns:
        QualityAnalyzerConfig: Default configuration object

    Examples:
        >>> config = get_default_config()
        >>> print(f"Sharpness weight: {config.weights.sharpness}")
        Sharpness weight: 0.6
    """
    return QualityAnalyzerConfig()


def create_custom_config(**kwargs) -> QualityAnalyzerConfig:
    """
    Create custom configuration by overriding specific values.

    Args:
        **kwargs: Configuration values to override

    Returns:
        QualityAnalyzerConfig: Custom configuration object

    Examples:
        >>> # Create config with custom weights
        >>> config = create_custom_config(
        ...     weights={'sharpness': 0.7, 'exposure': 0.3}
        ... )

        >>> # Create config with custom thresholds
        >>> config = create_custom_config(
        ...     thresholds={'high_quality_min': 75.0}
        ... )
    """
    config = QualityAnalyzerConfig()

    # Override weights if provided
    if 'weights' in kwargs:
        weights_dict = kwargs['weights']
        if 'sharpness' in weights_dict:
            config.weights.sharpness = weights_dict['sharpness']
        if 'exposure' in weights_dict:
            config.weights.exposure = weights_dict['exposure']
        if 'composition' in weights_dict:
            config.weights.composition = weights_dict['composition']

    # Override thresholds if provided
    if 'thresholds' in kwargs:
        thresholds_dict = kwargs['thresholds']
        for key, value in thresholds_dict.items():
            if hasattr(config.thresholds, key):
                setattr(config.thresholds, key, value)

    # Validate before returning
    config.validate()

    return config


# Preset configurations for different use cases

def get_conservative_config() -> QualityAnalyzerConfig:
    """
    Conservative configuration: Higher thresholds, stricter filtering.

    Use when you want only the highest quality photos.
    """
    return create_custom_config(
        thresholds={
            'high_quality_min': 80.0,
            'acceptable_min': 60.0
        }
    )


def get_permissive_config() -> QualityAnalyzerConfig:
    """
    Permissive configuration: Lower thresholds, more inclusive filtering.

    Use when you want to include more photos even if slightly lower quality.
    """
    return create_custom_config(
        thresholds={
            'high_quality_min': 60.0,
            'acceptable_min': 40.0,
            'low_quality_max': 39.0  # Must be less than acceptable_min
        }
    )


def get_sharpness_focused_config() -> QualityAnalyzerConfig:
    """
    Sharpness-focused configuration: Prioritize sharpness over exposure.

    Use when blur is the primary concern (e.g., action photography).
    """
    return create_custom_config(
        weights={
            'sharpness': 0.8,
            'exposure': 0.2
        }
    )


def get_exposure_focused_config() -> QualityAnalyzerConfig:
    """
    Exposure-focused configuration: Prioritize exposure over sharpness.

    Use when proper lighting is the primary concern (e.g., studio photography).
    """
    return create_custom_config(
        weights={
            'sharpness': 0.4,
            'exposure': 0.6
        }
    )
