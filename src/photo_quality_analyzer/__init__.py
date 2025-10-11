"""
Photo Quality Analyzer

A modular, high-performance system for analyzing photo quality based on
sharpness and exposure metrics.

Phase 1: Core Algorithm
- Sharpness detection using Laplacian variance
- Exposure analysis using histogram distribution
- Composite scoring with configurable weights

Phase 2: Infrastructure
- Batch processing with parallel execution
- Result caching with SQLite backend
- Photo library scanning
- Performance monitoring

Examples:
    >>> from photo_quality_analyzer import PhotoQualityAnalyzer
    >>> analyzer = PhotoQualityAnalyzer()
    >>> score = analyzer.analyze_photo('family_photo.jpg')
    >>> print(f"Quality: {score.composite:.1f} ({score.tier})")
    >>>
    >>> # Batch processing
    >>> from photo_quality_analyzer import BatchProcessor
    >>> processor = BatchProcessor()
    >>> result = processor.process_batch(photo_paths)
    >>> print(f"Analyzed {result.successful}/{result.total_photos} photos")
"""

__version__ = '2.0.0'
__author__ = 'Remember Twelve Team'

# Phase 1: Core Algorithm
from .analyzer import PhotoQualityAnalyzer, analyze_photo_simple
from .metrics.composite import QualityScore, create_quality_score
from .config import (
    QualityAnalyzerConfig,
    get_default_config,
    create_custom_config,
    get_conservative_config,
    get_permissive_config
)

# Phase 2: Infrastructure
from .batch_processor import BatchProcessor, BatchResult
from .cache import ResultCache
from .scanner import LibraryScanner, ScanStatistics
from .performance import (
    PerformanceMonitor,
    PerformanceMetrics,
    AggregateMonitor,
    track_performance
)

__all__ = [
    # Phase 1: Core Algorithm
    'PhotoQualityAnalyzer',
    'analyze_photo_simple',
    'QualityScore',
    'create_quality_score',
    'QualityAnalyzerConfig',
    'get_default_config',
    'create_custom_config',
    'get_conservative_config',
    'get_permissive_config',
    # Phase 2: Infrastructure
    'BatchProcessor',
    'BatchResult',
    'ResultCache',
    'LibraryScanner',
    'ScanStatistics',
    'PerformanceMonitor',
    'PerformanceMetrics',
    'AggregateMonitor',
    'track_performance',
]
