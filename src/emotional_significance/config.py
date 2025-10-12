"""
Emotional Significance Detector - Configuration

This module contains all configurable parameters for emotional significance detection.
Values are tuned based on empirical testing and OpenCV best practices.

Configuration includes:
- Face detection parameters (DNN model settings)
- Smile detection parameters (Haar Cascade settings)
- Scoring thresholds and weights
- Model file paths
"""

from typing import Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FaceDetectionConfig:
    """Configuration for DNN-based face detection."""

    # Detection confidence threshold
    confidence_threshold: float = 0.5  # Minimum confidence for detection (0.0-1.0)

    # Size filtering
    min_face_size_ratio: float = 0.05  # Min face size as % of image (5%)
    max_face_size_ratio: float = 0.95  # Max face size as % of image (95%)

    # Limits
    max_faces: int = 20  # Maximum faces to detect

    # DNN input parameters
    dnn_input_size: Tuple[int, int] = (300, 300)  # ResNet-10 SSD expects 300x300
    dnn_scale_factor: float = 1.0
    dnn_mean: Tuple[float, float, float] = (104.0, 177.0, 123.0)  # Caffe model mean


@dataclass
class SmileDetectionConfig:
    """Configuration for Haar Cascade smile detection."""

    # Haar Cascade parameters
    scale_factor: float = 1.1  # Scale reduction between detection passes (1.1-2.0)
    min_neighbors: int = 20  # Min neighbors for detection (higher = fewer false positives)

    # Smile size constraints (relative to face)
    min_smile_width_ratio: float = 0.3  # Smile should be at least 30% of face width
    min_smile_height_ratio: float = 0.2  # Smile should be at least 20% of face height

    # Confidence thresholds
    clear_smile_threshold: float = 0.8  # Strong/clear smile
    subtle_smile_threshold: float = 0.5  # Subtle smile


@dataclass
class ProximityConfig:
    """Configuration for intimacy/closeness calculation."""

    # Distance thresholds (normalized by average face width)
    # Distance = Euclidean distance between face centers / average face width

    very_close_threshold: float = 1.5  # Embracing, very intimate (85-100 points)
    close_threshold: float = 3.0  # Close together (70-84 points)
    moderate_threshold: float = 5.0  # Moderate distance (50-69 points)
    # > 5.0 = distant (0-49 points)

    # Score ranges for each tier
    very_close_score: float = 90.0
    close_score: float = 75.0
    moderate_score: float = 55.0
    distant_score: float = 25.0


@dataclass
class EngagementConfig:
    """Configuration for camera engagement detection."""

    # Frontal face aspect ratio range (width/height)
    frontal_ratio_min: float = 0.75  # Slightly wide faces
    frontal_ratio_max: float = 1.25  # Slightly tall faces

    # Profile face thresholds
    profile_ratio_threshold: float = 0.6  # < 0.6 or > 1.5 = likely profile

    # Engagement scoring
    frontal_score: float = 100.0  # Fully engaged (looking at camera)
    partial_score: float = 60.0  # Partially engaged
    profile_score: float = 20.0  # Profile view (not engaged)


@dataclass
class ScoringWeightsConfig:
    """Configuration for composite score weights."""

    # Component weights (sum to 100 points)
    face_presence_weight: float = 30.0  # 0-30 points
    emotion_weight: float = 40.0  # 0-40 points
    intimacy_weight: float = 20.0  # 0-20 points
    engagement_weight: float = 10.0  # 0-10 points

    def validate(self) -> bool:
        """Validate that weights sum to 100."""
        total = (self.face_presence_weight +
                 self.emotion_weight +
                 self.intimacy_weight +
                 self.engagement_weight)
        return 99.0 <= total <= 101.0  # Allow small floating point error

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'face_presence': self.face_presence_weight,
            'emotion': self.emotion_weight,
            'intimacy': self.intimacy_weight,
            'engagement': self.engagement_weight
        }


@dataclass
class TierThresholdsConfig:
    """Configuration for emotional significance tier thresholds."""

    # Composite score thresholds
    high_tier_min: float = 70.0  # Score >= 70: High significance
    medium_tier_min: float = 40.0  # Score >= 40: Medium significance
    # Score < 40: Low significance

    # Face count thresholds
    group_photo_threshold: int = 5  # >= 5 faces = group photo (bonus)
    couple_threshold: int = 2  # 2 faces = couple/pair


@dataclass
class ModelPathsConfig:
    """Configuration for OpenCV model file paths."""

    # Face detection DNN model (Caffe ResNet-10 SSD)
    face_model_prototxt: str = "deploy.prototxt"
    face_model_weights: str = "res10_300x300_ssd_iter_140000.caffemodel"

    # Smile detection Haar Cascade
    # Note: Built into OpenCV, loaded via cv2.data.haarcascades
    smile_cascade_name: str = "haarcascade_smile.xml"

    def get_face_model_paths(self, base_dir: Path) -> Tuple[Path, Path]:
        """
        Get full paths to face detection model files.

        Args:
            base_dir: Base directory containing model files

        Returns:
            Tuple of (prototxt_path, weights_path)
        """
        prototxt = base_dir / self.face_model_prototxt
        weights = base_dir / self.face_model_weights
        return prototxt, weights


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    # Image preprocessing
    max_image_size: int = 1024  # Resize to max 1024px (maintains quality)

    # Batch processing (for Phase 2)
    default_batch_size: int = 500
    default_num_workers: int = 4

    # Timeouts
    per_photo_timeout: int = 30  # Seconds per photo analysis


@dataclass
class EmotionalConfig:
    """
    Master configuration for Emotional Significance Detector.

    Contains all sub-configurations for different components.

    Examples:
        >>> # Use default configuration
        >>> config = EmotionalConfig()
        >>> config.validate()

        >>> # Custom configuration
        >>> config = EmotionalConfig(
        ...     face_detection=FaceDetectionConfig(confidence_threshold=0.6),
        ...     scoring_weights=ScoringWeightsConfig(emotion_weight=50.0)
        ... )
    """

    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    smile_detection: SmileDetectionConfig = field(default_factory=SmileDetectionConfig)
    proximity: ProximityConfig = field(default_factory=ProximityConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    scoring_weights: ScoringWeightsConfig = field(default_factory=ScoringWeightsConfig)
    tier_thresholds: TierThresholdsConfig = field(default_factory=TierThresholdsConfig)
    model_paths: ModelPathsConfig = field(default_factory=ModelPathsConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Algorithm version (for future upgrades)
    algorithm_version: str = "v1.0"

    def validate(self) -> bool:
        """
        Validate entire configuration.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.scoring_weights.validate():
            raise ValueError("Invalid scoring weights: must sum to 100")

        if self.tier_thresholds.high_tier_min <= self.tier_thresholds.medium_tier_min:
            raise ValueError("high_tier_min must be > medium_tier_min")

        if self.face_detection.confidence_threshold < 0 or self.face_detection.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

        return True

    def to_dict(self) -> Dict[str, any]:
        """Convert entire config to dictionary."""
        return {
            'face_detection': {
                'confidence_threshold': self.face_detection.confidence_threshold,
                'min_face_size_ratio': self.face_detection.min_face_size_ratio,
                'max_faces': self.face_detection.max_faces
            },
            'scoring_weights': self.scoring_weights.to_dict(),
            'tier_thresholds': {
                'high_tier_min': self.tier_thresholds.high_tier_min,
                'medium_tier_min': self.tier_thresholds.medium_tier_min
            },
            'algorithm_version': self.algorithm_version
        }


# Default configuration instance
DEFAULT_CONFIG = EmotionalConfig()


def get_default_config() -> EmotionalConfig:
    """
    Get default configuration.

    Returns:
        EmotionalConfig: Default configuration object

    Examples:
        >>> config = get_default_config()
        >>> print(f"Face confidence threshold: {config.face_detection.confidence_threshold}")
        Face confidence threshold: 0.5
    """
    return EmotionalConfig()


def create_custom_config(**kwargs) -> EmotionalConfig:
    """
    Create custom configuration by overriding specific values.

    Args:
        **kwargs: Configuration values to override

    Returns:
        EmotionalConfig: Custom configuration object

    Examples:
        >>> # Increase face detection confidence
        >>> config = create_custom_config(
        ...     face_detection={'confidence_threshold': 0.7}
        ... )

        >>> # Adjust emotion weight
        >>> config = create_custom_config(
        ...     scoring_weights={'emotion_weight': 50.0}
        ... )
    """
    config = EmotionalConfig()

    # Override face detection if provided
    if 'face_detection' in kwargs:
        fd_dict = kwargs['face_detection']
        for key, value in fd_dict.items():
            if hasattr(config.face_detection, key):
                setattr(config.face_detection, key, value)

    # Override scoring weights if provided
    if 'scoring_weights' in kwargs:
        sw_dict = kwargs['scoring_weights']
        for key, value in sw_dict.items():
            if hasattr(config.scoring_weights, key):
                setattr(config.scoring_weights, key, value)

    # Override tier thresholds if provided
    if 'tier_thresholds' in kwargs:
        tt_dict = kwargs['tier_thresholds']
        for key, value in tt_dict.items():
            if hasattr(config.tier_thresholds, key):
                setattr(config.tier_thresholds, key, value)

    # Validate before returning
    config.validate()

    return config


# Preset configurations

def get_conservative_config() -> EmotionalConfig:
    """
    Conservative configuration: Higher thresholds, stricter detection.

    Use when you want only the most emotionally significant photos.
    """
    return create_custom_config(
        face_detection={'confidence_threshold': 0.7},
        tier_thresholds={'high_tier_min': 80.0, 'medium_tier_min': 50.0}
    )


def get_permissive_config() -> EmotionalConfig:
    """
    Permissive configuration: Lower thresholds, more inclusive detection.

    Use when you want to include more photos with emotional content.
    """
    return create_custom_config(
        face_detection={'confidence_threshold': 0.4},
        tier_thresholds={'high_tier_min': 60.0, 'medium_tier_min': 30.0}
    )


def get_emotion_focused_config() -> EmotionalConfig:
    """
    Emotion-focused configuration: Prioritize smiles and positive emotions.

    Use when capturing happiness and joy is the primary goal.
    """
    return create_custom_config(
        scoring_weights={
            'face_presence_weight': 20.0,
            'emotion_weight': 50.0,
            'intimacy_weight': 20.0,
            'engagement_weight': 10.0
        }
    )


def get_intimacy_focused_config() -> EmotionalConfig:
    """
    Intimacy-focused configuration: Prioritize physical closeness.

    Use when capturing close relationships and connections is the goal.
    """
    return create_custom_config(
        scoring_weights={
            'face_presence_weight': 20.0,
            'emotion_weight': 30.0,
            'intimacy_weight': 40.0,
            'engagement_weight': 10.0
        }
    )
