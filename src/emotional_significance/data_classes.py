"""
Emotional Significance Detector - Data Classes

Core data structures for face detection and emotional scoring.

This module defines the fundamental data types used throughout the
emotional significance detection system:
- FaceDetection: Information about detected faces
- EmotionalScore: Complete emotional significance assessment
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional


@dataclass
class FaceDetection:
    """
    Information about a detected face.

    Attributes:
        bbox: Bounding box (x, y, w, h) in pixels
        confidence: Detection confidence (0.0-1.0)
        center: Face center point (x, y)
        size_ratio: Face size relative to image (0.0-1.0)
        smile_confidence: Smile detection confidence (0.0-1.0, None if not detected)
        landmarks: Optional facial landmarks (for advanced detection)

    Examples:
        >>> face = FaceDetection(
        ...     bbox=(100, 100, 200, 250),
        ...     confidence=0.95,
        ...     center=(200, 225),
        ...     size_ratio=0.15,
        ...     smile_confidence=0.8
        ... )
        >>> print(f"Face area: {face.area} pixels")
        >>> print(f"Is smiling: {face.is_smiling}")
    """
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    center: Tuple[int, int]
    size_ratio: float
    smile_confidence: Optional[float] = None
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None

    @property
    def area(self) -> int:
        """Calculate face area in pixels."""
        return self.bbox[2] * self.bbox[3]

    @property
    def is_smiling(self) -> bool:
        """Check if face has confident smile (>0.5)."""
        return self.smile_confidence is not None and self.smile_confidence > 0.5

    @property
    def width(self) -> int:
        """Get face width."""
        return self.bbox[2]

    @property
    def height(self) -> int:
        """Get face height."""
        return self.bbox[3]

    @property
    def aspect_ratio(self) -> float:
        """Calculate face aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'center': self.center,
            'size_ratio': self.size_ratio,
            'smile_confidence': self.smile_confidence,
            'landmarks': self.landmarks,
            'area': self.area,
            'is_smiling': self.is_smiling
        }


@dataclass
class EmotionalScore:
    """
    Emotional significance score for a photo.

    Combines multiple signals (faces, emotions, intimacy, engagement) into
    a comprehensive assessment of a photo's emotional significance.

    Attributes:
        face_count: Number of faces detected (0-20+)
        face_coverage: Percentage of image covered by faces (0.0-1.0)
        emotion_score: Positive emotion score (0-100)
        intimacy_score: Physical closeness score (0-100)
        engagement_score: Camera engagement score (0-100)
        composite: Overall emotional significance (0-100)
        tier: Emotional significance tier ('high', 'medium', 'low')
        metadata: Additional detection details

    Score Components (sum to 100):
        - Face Presence: 0-30 points (based on count and coverage)
        - Emotion: 0-40 points (smiles and positive expressions)
        - Intimacy: 0-20 points (physical closeness between people)
        - Engagement: 0-10 points (faces looking at camera)

    Tiers:
        - High: 70-100 (memorable moments worth prioritizing)
        - Medium: 40-69 (decent emotional content)
        - Low: 0-39 (minimal emotional significance)

    Examples:
        >>> score = EmotionalScore(
        ...     face_count=2,
        ...     face_coverage=0.35,
        ...     emotion_score=85.0,
        ...     intimacy_score=90.0,
        ...     engagement_score=80.0,
        ...     composite=88.0,
        ...     tier='high',
        ...     metadata={'smiling_faces': 2}
        ... )
        >>> print(score)
        EmotionalScore(composite=88.0, faces=2, emotion=85.0, tier='high')
        >>> print(f"This is a {score.tier} significance photo")
    """
    face_count: int
    face_coverage: float
    emotion_score: float
    intimacy_score: float
    engagement_score: float
    composite: float
    tier: str
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        """Human-readable representation."""
        return (f"EmotionalScore(composite={self.composite:.1f}, "
                f"faces={self.face_count}, emotion={self.emotion_score:.1f}, "
                f"tier='{self.tier}')")

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"EmotionalScore(face_count={self.face_count}, "
                f"face_coverage={self.face_coverage:.2f}, "
                f"emotion={self.emotion_score:.1f}, "
                f"intimacy={self.intimacy_score:.1f}, "
                f"engagement={self.engagement_score:.1f}, "
                f"composite={self.composite:.1f}, "
                f"tier='{self.tier}')")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary with all score components and metadata

        Examples:
            >>> score_dict = score.to_dict()
            >>> import json
            >>> json_str = json.dumps(score_dict)
        """
        return {
            'face_count': self.face_count,
            'face_coverage': self.face_coverage,
            'emotion_score': self.emotion_score,
            'intimacy_score': self.intimacy_score,
            'engagement_score': self.engagement_score,
            'composite': self.composite,
            'tier': self.tier,
            'metadata': self.metadata
        }

    @property
    def is_high_significance(self) -> bool:
        """Check if photo has high emotional significance."""
        return self.tier == 'high'

    @property
    def is_medium_significance(self) -> bool:
        """Check if photo has medium emotional significance."""
        return self.tier == 'medium'

    @property
    def is_low_significance(self) -> bool:
        """Check if photo has low emotional significance."""
        return self.tier == 'low'

    @property
    def has_faces(self) -> bool:
        """Check if photo contains any faces."""
        return self.face_count > 0

    @property
    def has_multiple_people(self) -> bool:
        """Check if photo contains multiple people."""
        return self.face_count >= 2

    @property
    def has_positive_emotion(self) -> bool:
        """Check if photo contains positive emotions (emotion score > 50)."""
        return self.emotion_score > 50.0
