"""
Emotional Significance Detector - Phase 2

A comprehensive system for detecting and scoring emotional significance in photos.
Analyzes faces, smiles, physical proximity, and camera engagement to identify
memorable moments worth curating.

Main Components:
- EmotionalAnalyzer: Primary interface for photo analysis
- EmotionalBatchProcessor: Parallel batch processing with progress tracking
- EmotionalResultCache: SQLite-based caching for analyzed photos
- FaceDetector: DNN-based face detection (ResNet-10 SSD)
- SmileDetector: Haar Cascade smile detection
- ProximityCalculator: Physical closeness/intimacy analysis
- EngagementDetector: Face orientation/camera engagement

Data Classes:
- FaceDetection: Information about detected faces
- EmotionalScore: Comprehensive emotional significance assessment
- BatchResult: Results from batch processing operations

Configuration:
- EmotionalConfig: Master configuration with all parameters
- Pre-configured presets: conservative, permissive, emotion-focused, intimacy-focused

Performance:
- Target: <20ms per photo (1024px) with caching
- Throughput: 50+ photos/sec with parallel processing
- Accuracy: >95% face detection, ~80% smile detection
- Local processing: No cloud APIs, privacy-preserving

Usage:
    Basic analysis:
    >>> from emotional_significance import EmotionalAnalyzer
    >>> analyzer = EmotionalAnalyzer()
    >>> score = analyzer.analyze_photo('photo.jpg')
    >>> print(f"Emotional significance: {score.composite:.1f} ({score.tier})")

    Batch analysis with caching:
    >>> from emotional_significance import EmotionalAnalyzer, EmotionalResultCache
    >>> cache = EmotionalResultCache('emotional_scores.db')
    >>> analyzer = EmotionalAnalyzer()
    >>>
    >>> photos = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
    >>> for photo in photos:
    ...     if cache.should_analyze(photo):
    ...         score = analyzer.analyze_photo(photo)
    ...         cache.set(photo, score)
    ...     else:
    ...         score = cache.get(photo)

    Parallel batch processing:
    >>> from emotional_significance import EmotionalBatchProcessor
    >>> processor = EmotionalBatchProcessor(num_workers=4)
    >>> result = processor.process_batch(photo_paths)
    >>> print(f"Analyzed {result.successful}/{result.total_photos} photos")

Version: 2.0.0 (Phase 2: Infrastructure Integration)
"""

__version__ = '2.0.0'
__author__ = 'Remember Twelve Team'

# Main analyzer
from .analyzer import EmotionalAnalyzer, analyze_photo_simple

# Phase 2: Infrastructure
from .cache import EmotionalResultCache
from .batch_processor import EmotionalBatchProcessor, BatchResult

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

    # Phase 2: Infrastructure
    'EmotionalResultCache',
    'EmotionalBatchProcessor',
    'BatchResult',

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
