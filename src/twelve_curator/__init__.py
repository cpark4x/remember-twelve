"""
Twelve Curation Engine

Automatically selects the best 12 photos from a year's library based on:
- Quality scores (sharpness + exposure)
- Emotional significance (faces + emotions)
- Temporal diversity (spread across months)
- Visual diversity (avoid near-duplicates)

Main Components:
- TwelveCurator: Main curation engine
- PhotoCandidate: Photo being considered for selection
- TwelveSelection: Final curated 12 photos
- CurationConfig: Configuration for curation strategies

Utilities:
- exif_utils: Extract datetime from EXIF metadata
- diversity: Visual similarity detection

Examples:
    Basic curation:
    >>> from twelve_curator import TwelveCurator
    >>> curator = TwelveCurator()
    >>> selection = curator.curate_year(photo_paths, year=2024)
    >>> print(selection.summary())

    Custom strategy:
    >>> from twelve_curator import TwelveCurator, CurationConfig
    >>> config = CurationConfig.people_first()
    >>> curator = TwelveCurator(config)
    >>> selection = curator.curate_year(photo_paths, year=2024)
"""

__version__ = '1.0.0'
__author__ = 'Remember Twelve Team'

# Main components
from .data_classes import PhotoCandidate, TwelveSelection, CurationConfig
from .curator import TwelveCurator

# Utilities (for advanced usage)
from . import exif_utils
from . import diversity

__all__ = [
    # Main interface
    'TwelveCurator',
    'PhotoCandidate',
    'TwelveSelection',
    'CurationConfig',

    # Utilities
    'exif_utils',
    'diversity',
]
