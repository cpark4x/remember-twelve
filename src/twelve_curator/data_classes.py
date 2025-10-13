"""
Data classes for the Twelve Curation Engine.

Defines the core data structures for photo candidates, final selections,
and curation configuration.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json


@dataclass
class PhotoCandidate:
    """
    A photo being considered for the Twelve.

    Attributes:
        photo_path: Path to the photo file
        timestamp: When the photo was taken
        month: Month (1-12) when photo was taken
        quality_score: Technical quality score (0-100)
        emotional_score: Emotional significance score (0-100)
        combined_score: Weighted combination of quality + emotional (0-100)
        metadata: Additional metadata (e.g., reasons, diversity info)
    """
    photo_path: Path
    timestamp: Optional[datetime]
    month: Optional[int]  # 1-12, None if unknown
    quality_score: float  # 0-100
    emotional_score: float  # 0-100
    combined_score: float  # 0-100
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            'photo_path': str(self.photo_path),
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'month': self.month,
            'quality_score': round(self.quality_score, 2),
            'emotional_score': round(self.emotional_score, 2),
            'combined_score': round(self.combined_score, 2),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PhotoCandidate':
        """Create from dictionary."""
        return cls(
            photo_path=Path(data['photo_path']),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None,
            month=data.get('month'),
            quality_score=data['quality_score'],
            emotional_score=data['emotional_score'],
            combined_score=data['combined_score'],
            metadata=data.get('metadata', {})
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        month_str = f"Month {self.month}" if self.month else "Unknown month"
        return (
            f"PhotoCandidate({self.photo_path.name}, {month_str}, "
            f"Combined: {self.combined_score:.1f}, "
            f"Q: {self.quality_score:.1f}, E: {self.emotional_score:.1f})"
        )


@dataclass
class TwelveSelection:
    """
    The final curated Twelve for a year.

    Attributes:
        year: Year being curated
        photos: Exactly 12 photos (or fewer if <12 available)
        strategy: Curation strategy used
        stats: Statistics about the selection
        created_at: When this curation was performed
    """
    year: int
    photos: List[PhotoCandidate]  # Should be exactly 12
    strategy: str
    stats: Dict
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate the selection."""
        if len(self.photos) > 12:
            raise ValueError(f"TwelveSelection cannot have more than 12 photos (got {len(self.photos)})")

    def to_dict(self) -> dict:
        """Export to dictionary (JSON-serializable)."""
        return {
            'year': self.year,
            'strategy': self.strategy,
            'created_at': self.created_at.isoformat(),
            'stats': self.stats,
            'photos': [photo.to_dict() for photo in self.photos]
        }

    def save(self, output_path: Path):
        """
        Save selection to JSON file.

        Args:
            output_path: Path to save the JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, input_path: Path) -> 'TwelveSelection':
        """
        Load selection from JSON file.

        Args:
            input_path: Path to the JSON file

        Returns:
            TwelveSelection instance
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        return cls(
            year=data['year'],
            photos=[PhotoCandidate.from_dict(p) for p in data['photos']],
            strategy=data['strategy'],
            stats=data['stats'],
            created_at=datetime.fromisoformat(data['created_at'])
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"TwelveSelection(year={self.year}, photos={len(self.photos)}, "
            f"strategy={self.strategy}, avg_score={self.stats.get('avg_combined', 0):.1f})"
        )

    def summary(self) -> str:
        """Generate a detailed summary of the selection."""
        lines = [
            f"\n{'='*60}",
            f"TWELVE SELECTION FOR {self.year}",
            f"{'='*60}",
            f"Strategy: {self.strategy}",
            f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nSTATISTICS:",
            f"  Total candidates: {self.stats.get('total_candidates', 0)}",
            f"  Photos selected: {len(self.photos)}",
            f"  Avg quality score: {self.stats.get('avg_quality', 0):.1f}",
            f"  Avg emotional score: {self.stats.get('avg_emotional', 0):.1f}",
            f"  Avg combined score: {self.stats.get('avg_combined', 0):.1f}",
            f"  Months represented: {self.stats.get('months_represented', 0)}",
            f"  Photos with faces: {self.stats.get('photos_with_faces', 0)}",
            f"\nSELECTED PHOTOS:",
        ]

        for i, photo in enumerate(self.photos, 1):
            month_str = f"Month {photo.month:02d}" if photo.month else "Unknown"
            lines.append(
                f"  {i:2d}. {month_str} | {photo.photo_path.name:40s} | "
                f"Score: {photo.combined_score:5.1f} (Q:{photo.quality_score:5.1f} E:{photo.emotional_score:5.1f})"
            )

        lines.append(f"{'='*60}\n")
        return '\n'.join(lines)


@dataclass
class CurationConfig:
    """
    Configuration for photo curation.

    Attributes:
        quality_weight: Weight for quality score (0-1)
        emotional_weight: Weight for emotional score (0-1)
        strategy: Curation strategy name
        visual_diversity_threshold: Similarity threshold for diversity (0-1)
        min_combined_score: Minimum score to consider (0-100)
        enforce_monthly_distribution: Whether to enforce one photo per month
        min_months_represented: Minimum months to represent (if available)
        enable_diversity_filter: Whether to apply visual diversity filter
        prefer_with_faces: Whether to prefer photos with faces
    """
    # Scoring weights
    quality_weight: float = 0.4
    emotional_weight: float = 0.6

    # Strategy
    strategy: str = "balanced"

    # Diversity settings
    visual_diversity_threshold: float = 0.85  # Photos more similar than this are too similar
    enable_diversity_filter: bool = True

    # Quality thresholds
    min_combined_score: float = 0.0  # Accept all by default

    # Temporal settings
    enforce_monthly_distribution: bool = True
    min_months_represented: int = 10

    # Preferences
    prefer_with_faces: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not (0 <= self.quality_weight <= 1):
            raise ValueError(f"quality_weight must be between 0 and 1 (got {self.quality_weight})")
        if not (0 <= self.emotional_weight <= 1):
            raise ValueError(f"emotional_weight must be between 0 and 1 (got {self.emotional_weight})")
        if abs(self.quality_weight + self.emotional_weight - 1.0) > 0.01:
            raise ValueError(
                f"quality_weight + emotional_weight must equal 1.0 "
                f"(got {self.quality_weight + self.emotional_weight})"
            )
        if not (0 <= self.visual_diversity_threshold <= 1):
            raise ValueError(
                f"visual_diversity_threshold must be between 0 and 1 "
                f"(got {self.visual_diversity_threshold})"
            )

    @classmethod
    def balanced(cls) -> 'CurationConfig':
        """Balanced strategy (default): 40% quality, 60% emotional."""
        return cls(
            quality_weight=0.4,
            emotional_weight=0.6,
            strategy="balanced"
        )

    @classmethod
    def aesthetic_first(cls) -> 'CurationConfig':
        """Aesthetic-first strategy: 70% quality, 30% emotional."""
        return cls(
            quality_weight=0.7,
            emotional_weight=0.3,
            strategy="aesthetic_first"
        )

    @classmethod
    def people_first(cls) -> 'CurationConfig':
        """People-first strategy: 30% quality, 70% emotional."""
        return cls(
            quality_weight=0.3,
            emotional_weight=0.7,
            strategy="people_first",
            prefer_with_faces=True
        )

    @classmethod
    def top_heavy(cls) -> 'CurationConfig':
        """Top-heavy strategy: Less temporal diversity, best scores."""
        return cls(
            quality_weight=0.4,
            emotional_weight=0.6,
            strategy="top_heavy",
            enforce_monthly_distribution=False,
            min_months_represented=6
        )

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return asdict(self)
