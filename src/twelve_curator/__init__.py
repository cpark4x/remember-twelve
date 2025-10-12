"""
Twelve Curation Engine

Automatically selects the best 12 photos from a year's library based on:
- Quality scores (sharpness + exposure)
- Emotional significance (faces + emotions)
- Temporal diversity (spread across months)
- Visual diversity (avoid near-duplicates)
"""

from .data_classes import PhotoCandidate, TwelveSelection, CurationConfig
from .curator import TwelveCurator

__all__ = [
    'PhotoCandidate',
    'TwelveSelection',
    'CurationConfig',
    'TwelveCurator',
]

__version__ = '1.0.0'
