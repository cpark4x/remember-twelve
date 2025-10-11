"""
Photo Quality Analyzer

A modular, high-performance system for analyzing photo quality based on
sharpness and exposure metrics.

Phase 1: Core Algorithm
- Sharpness detection using Laplacian variance
- Exposure analysis using histogram distribution
- Composite scoring with configurable weights

Examples:
    >>> from photo_quality_analyzer import PhotoQualityAnalyzer
    >>> analyzer = PhotoQualityAnalyzer()
    >>> score = analyzer.analyze_photo('family_photo.jpg')
    >>> print(f"Quality: {score.composite:.1f} ({score.tier})")
"""

__version__ = '1.0.0'
__author__ = 'Remember Twelve Team'

from .analyzer import PhotoQualityAnalyzer, analyze_photo_simple
from .metrics.composite import QualityScore, create_quality_score
from .config import (
    QualityAnalyzerConfig,
    get_default_config,
    create_custom_config,
    get_conservative_config,
    get_permissive_config
)

__all__ = [
    'PhotoQualityAnalyzer',
    'analyze_photo_simple',
    'QualityScore',
    'create_quality_score',
    'QualityAnalyzerConfig',
    'get_default_config',
    'create_custom_config',
    'get_conservative_config',
    'get_permissive_config',
]
