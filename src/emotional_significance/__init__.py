"""
Emotional Significance Detector - Phase 1

A comprehensive system for detecting and scoring emotional significance in photos.
Analyzes faces, smiles, physical proximity, and camera engagement to identify
memorable moments worth curating.

Main Components:
- EmotionalAnalyzer: Primary interface for photo analysis
- FaceDetector: DNN-based face detection (ResNet-10 SSD)
- SmileDetector: Haar Cascade smile detection
- ProximityCalculator: Physical closeness/intimacy analysis
- EngagementDetector: Face orientation/camera engagement

Data Classes:
- FaceDetection: Information about detected faces
- EmotionalScore: Comprehensive emotional significance assessment

Configuration:
- EmotionalConfig: Master configuration with all parameters
- Pre-configured presets: conservative, permissive, emotion-focused, intimacy-focused

Performance:
- Target: <50ms per photo (1024px)
- Accuracy: >95% face detection, ~80% smile detection
- Local processing: No cloud APIs, privacy-preserving

Usage:
    Basic analysis:
    >>> from emotional_significance import EmotionalAnalyzer
    >>> analyzer = EmotionalAnalyzer()
    >>> score = analyzer.analyze_photo('photo.jpg')
    >>> print(f"Emotional significance: {score.composite:.1f} ({score.tier})")

    Custom configuration:
    >>> from emotional_significance import EmotionalAnalyzer, create_custom_config
    >>> config = create_custom_config(
    ...     face_detection={'confidence_threshold': 0.7}
    ... )
    >>> analyzer = EmotionalAnalyzer(config=config)

    Batch analysis:
    >>> photos = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
    >>> scores = analyzer.analyze_batch(photos)
    >>> high_sig_photos = [p for p, s in zip(photos, scores)
    ...                    if s and s.tier == 'high']

Version: 1.0.0 (Phase 1: Core Detection)
"""

__version__ = '1.0.0'
__author__ = 'Remember Twelve Team'

# Main analyzer
from .analyzer import EmotionalAnalyzer, analyze_photo_simple

# Data classes
from .data_classes import FaceDetection, EmotionalScore

# Configuration
from .config import (
    EmotionalConfig,
    get_default_config,
    create_custom_config,
    get_conservative_config,
    get_permissive_config,
    get_emotion_focused_config,
    get_intimacy_focused_config
)

# Detectors (for advanced usage)
from .detectors import (
    FaceDetector,
    SmileDetector,
    ProximityCalculator,
    EngagementDetector
)

# Scoring utilities (for advanced usage)
from .scoring import (
    create_emotional_score,
    calculate_face_presence_score,
    calculate_emotion_score,
    calculate_intimacy_score_component,
    calculate_engagement_score_component
)

__all__ = [
    # Main interface
    'EmotionalAnalyzer',
    'analyze_photo_simple',

    # Data classes
    'FaceDetection',
    'EmotionalScore',

    # Configuration
    'EmotionalConfig',
    'get_default_config',
    'create_custom_config',
    'get_conservative_config',
    'get_permissive_config',
    'get_emotion_focused_config',
    'get_intimacy_focused_config',

    # Detectors
    'FaceDetector',
    'SmileDetector',
    'ProximityCalculator',
    'EngagementDetector',

    # Scoring
    'create_emotional_score',
    'calculate_face_presence_score',
    'calculate_emotion_score',
    'calculate_intimacy_score_component',
    'calculate_engagement_score_component',
]
