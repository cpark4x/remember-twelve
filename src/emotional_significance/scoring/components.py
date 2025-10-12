"""
Scoring Components - Individual Score Calculations

This module implements the individual scoring components that feed into
the composite emotional significance score:

1. Face Presence Score (0-30 points):
   - Based on number of faces and face coverage
   - More faces = higher score (up to a point)

2. Emotion Score (0-40 points):
   - Based on number of smiling faces and smile intensity
   - More smiles = higher score

3. Intimacy Score Component (0-20 points):
   - Based on physical closeness between faces
   - Closer faces = higher score

4. Engagement Score Component (0-10 points):
   - Based on faces looking at camera (frontal vs profile)
   - More frontal faces = higher score
"""

from typing import List
import logging

from ..data_classes import FaceDetection
from ..config import ScoringWeightsConfig

logger = logging.getLogger(__name__)


def calculate_face_presence_score(face_count: int,
                                  face_coverage: float,
                                  max_score: float = 30.0) -> float:
    """
    Calculate face presence score (0-30 points).

    Rewards photos with people in them. More faces and greater coverage
    increase the score.

    Scoring:
    - 0 faces: 0 points
    - 1 face: 10-20 points (based on coverage)
    - 2 faces: 20-25 points
    - 3-5 faces: 25-28 points
    - 6+ faces: 28-30 points (group photo bonus)

    Args:
        face_count: Number of detected faces
        face_coverage: Percentage of image covered by faces (0.0-1.0)
        max_score: Maximum possible score (default 30.0)

    Returns:
        Face presence score (0-max_score)

    Examples:
        >>> # Single person, large face
        >>> score = calculate_face_presence_score(1, 0.4)
        >>> print(f"Score: {score:.1f}/30")

        >>> # Group photo
        >>> score = calculate_face_presence_score(6, 0.6)
        >>> print(f"Score: {score:.1f}/30")
    """
    if face_count == 0:
        return 0.0

    # Base score from face count
    if face_count == 1:
        # Single person: 10-20 points based on coverage
        base_score = 10.0 + (face_coverage * 10.0)
    elif face_count == 2:
        # Couple/pair: 20-25 points
        base_score = 20.0 + (face_coverage * 5.0)
    elif face_count <= 5:
        # Small group: 25-28 points
        base_score = 25.0 + (face_coverage * 3.0)
    else:
        # Large group: 28-30 points (bonus for group photos)
        base_score = 28.0 + min(2.0, (face_count - 5) * 0.5)

    # Clamp to max_score
    score = min(max_score, base_score)

    return float(score)


def calculate_emotion_score(faces: List[FaceDetection],
                           max_score: float = 40.0) -> float:
    """
    Calculate emotion score (0-40 points).

    Rewards positive emotions (smiles). More smiling faces and stronger
    smiles increase the score.

    Scoring:
    - No smiles: 0-5 points (neutral)
    - Some smiles: 15-30 points (based on count and intensity)
    - Many smiles: 30-40 points (joyful group)

    Args:
        faces: List of FaceDetection objects with smile_confidence
        max_score: Maximum possible score (default 40.0)

    Returns:
        Emotion score (0-max_score)

    Examples:
        >>> # Everyone smiling
        >>> faces = [FaceDetection(..., smile_confidence=0.9) for _ in range(3)]
        >>> score = calculate_emotion_score(faces)
        >>> print(f"Score: {score:.1f}/40")
    """
    if not faces:
        return 0.0

    # Count smiling faces and calculate average smile confidence
    smiling_faces = [f for f in faces if f.is_smiling]
    num_smiling = len(smiling_faces)

    if num_smiling == 0:
        # No smiles: minimal score
        return 5.0

    # Calculate average smile intensity
    avg_smile_confidence = sum(f.smile_confidence for f in smiling_faces) / num_smiling

    # Calculate ratio of smiling faces
    smile_ratio = num_smiling / len(faces)

    # Base score from smile ratio
    base_score = smile_ratio * 30.0  # 0-30 points

    # Bonus for high smile intensity
    intensity_bonus = avg_smile_confidence * 10.0  # 0-10 points

    # Total score
    total_score = base_score + intensity_bonus

    # Clamp to max_score
    score = min(max_score, total_score)

    return float(score)


def calculate_intimacy_score_component(intimacy_raw_score: float,
                                      max_score: float = 20.0) -> float:
    """
    Calculate intimacy score component (0-20 points).

    Converts raw intimacy score (0-100) to component score (0-20).

    Args:
        intimacy_raw_score: Raw intimacy score from ProximityCalculator (0-100)
        max_score: Maximum possible score (default 20.0)

    Returns:
        Intimacy component score (0-max_score)

    Examples:
        >>> # High intimacy (close faces)
        >>> score = calculate_intimacy_score_component(85.0)
        >>> print(f"Score: {score:.1f}/20")
    """
    # Simple linear scaling
    score = (intimacy_raw_score / 100.0) * max_score

    return float(score)


def calculate_engagement_score_component(engagement_raw_score: float,
                                        max_score: float = 10.0) -> float:
    """
    Calculate engagement score component (0-10 points).

    Converts raw engagement score (0-100) to component score (0-10).

    Args:
        engagement_raw_score: Raw engagement score from EngagementDetector (0-100)
        max_score: Maximum possible score (default 10.0)

    Returns:
        Engagement component score (0-max_score)

    Examples:
        >>> # High engagement (all frontal)
        >>> score = calculate_engagement_score_component(95.0)
        >>> print(f"Score: {score:.1f}/10")
    """
    # Simple linear scaling
    score = (engagement_raw_score / 100.0) * max_score

    return float(score)


def validate_component_scores(face_presence: float,
                              emotion: float,
                              intimacy: float,
                              engagement: float,
                              weights: ScoringWeightsConfig) -> bool:
    """
    Validate that component scores are within expected ranges.

    Args:
        face_presence: Face presence score
        emotion: Emotion score
        intimacy: Intimacy score
        engagement: Engagement score
        weights: Scoring weights configuration

    Returns:
        True if all scores are valid

    Raises:
        ValueError: If any score is out of range
    """
    if not (0 <= face_presence <= weights.face_presence_weight):
        raise ValueError(f"Face presence score {face_presence} exceeds max {weights.face_presence_weight}")

    if not (0 <= emotion <= weights.emotion_weight):
        raise ValueError(f"Emotion score {emotion} exceeds max {weights.emotion_weight}")

    if not (0 <= intimacy <= weights.intimacy_weight):
        raise ValueError(f"Intimacy score {intimacy} exceeds max {weights.intimacy_weight}")

    if not (0 <= engagement <= weights.engagement_weight):
        raise ValueError(f"Engagement score {engagement} exceeds max {weights.engagement_weight}")

    return True
