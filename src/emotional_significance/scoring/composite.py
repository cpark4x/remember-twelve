"""
Composite Scoring - EmotionalScore Creation

This module combines individual score components into a final EmotionalScore
object with composite score and tier classification.

Composite Score (0-100):
- Face Presence: 0-30 points
- Emotion: 0-40 points
- Intimacy: 0-20 points
- Engagement: 0-10 points
- Total: 100 points

Tiers:
- High: 70-100 (memorable moments worth prioritizing)
- Medium: 40-69 (decent emotional content)
- Low: 0-39 (minimal emotional significance)
"""

from typing import Dict, Any, List
import logging

from ..data_classes import EmotionalScore, FaceDetection
from ..config import TierThresholdsConfig, ScoringWeightsConfig
from .components import (
    calculate_face_presence_score,
    calculate_emotion_score,
    calculate_intimacy_score_component,
    calculate_engagement_score_component,
    validate_component_scores
)

logger = logging.getLogger(__name__)


def create_emotional_score(
    faces: List[FaceDetection],
    face_coverage: float,
    intimacy_raw_score: float,
    engagement_raw_score: float,
    weights: ScoringWeightsConfig,
    thresholds: TierThresholdsConfig,
    metadata: Dict[str, Any]
) -> EmotionalScore:
    """
    Create comprehensive EmotionalScore from component scores.

    Args:
        faces: List of detected faces with smile information
        face_coverage: Percentage of image covered by faces (0.0-1.0)
        intimacy_raw_score: Raw intimacy score from ProximityCalculator (0-100)
        engagement_raw_score: Raw engagement score from EngagementDetector (0-100)
        weights: Scoring weights configuration
        thresholds: Tier thresholds configuration
        metadata: Additional detection metadata

    Returns:
        EmotionalScore object with all components and composite score

    Examples:
        >>> score = create_emotional_score(
        ...     faces=faces,
        ...     face_coverage=0.4,
        ...     intimacy_raw_score=85.0,
        ...     engagement_raw_score=90.0,
        ...     weights=config.scoring_weights,
        ...     thresholds=config.tier_thresholds,
        ...     metadata={}
        ... )
        >>> print(f"Composite: {score.composite:.1f}")
    """
    # Calculate individual component scores
    face_presence = calculate_face_presence_score(
        len(faces),
        face_coverage,
        weights.face_presence_weight
    )

    emotion = calculate_emotion_score(
        faces,
        weights.emotion_weight
    )

    intimacy = calculate_intimacy_score_component(
        intimacy_raw_score,
        weights.intimacy_weight
    )

    engagement = calculate_engagement_score_component(
        engagement_raw_score,
        weights.engagement_weight
    )

    # Validate component scores
    try:
        validate_component_scores(face_presence, emotion, intimacy, engagement, weights)
    except ValueError as e:
        logger.warning(f"Component score validation failed: {e}")
        # Continue anyway, but log the issue

    # Calculate composite score
    composite = face_presence + emotion + intimacy + engagement

    # Clamp to 0-100 range (should already be in range, but safety check)
    composite = max(0.0, min(100.0, composite))

    # Determine tier
    tier = determine_tier(composite, thresholds)

    # Enhance metadata with component scores
    enhanced_metadata = {
        **metadata,
        'face_presence_score': float(face_presence),
        'emotion_score_component': float(emotion),
        'intimacy_score_component': float(intimacy),
        'engagement_score_component': float(engagement)
    }

    # Create EmotionalScore object
    score = EmotionalScore(
        face_count=len(faces),
        face_coverage=float(face_coverage),
        emotion_score=float(emotion),  # Note: This is the component score (0-40)
        intimacy_score=float(intimacy_raw_score),  # Store raw score (0-100) for reference
        engagement_score=float(engagement_raw_score),  # Store raw score (0-100) for reference
        composite=float(composite),
        tier=tier,
        metadata=enhanced_metadata
    )

    logger.debug(f"Created EmotionalScore: {score}")

    return score


def determine_tier(composite_score: float,
                  thresholds: TierThresholdsConfig) -> str:
    """
    Determine emotional significance tier from composite score.

    Args:
        composite_score: Composite score (0-100)
        thresholds: Tier thresholds configuration

    Returns:
        Tier string: 'high', 'medium', or 'low'

    Examples:
        >>> tier = determine_tier(75.0, config.tier_thresholds)
        >>> print(tier)  # 'high'
    """
    if composite_score >= thresholds.high_tier_min:
        return 'high'
    elif composite_score >= thresholds.medium_tier_min:
        return 'medium'
    else:
        return 'low'


def calculate_composite_simple(face_presence: float,
                               emotion: float,
                               intimacy: float,
                               engagement: float) -> float:
    """
    Simple composite calculation (sum of components).

    Useful for testing or custom calculations.

    Args:
        face_presence: Face presence score (0-30)
        emotion: Emotion score (0-40)
        intimacy: Intimacy score (0-20)
        engagement: Engagement score (0-10)

    Returns:
        Composite score (0-100)
    """
    composite = face_presence + emotion + intimacy + engagement
    return max(0.0, min(100.0, composite))


def get_score_breakdown(score: EmotionalScore) -> Dict[str, Any]:
    """
    Get detailed breakdown of score components.

    Args:
        score: EmotionalScore object

    Returns:
        Dictionary with detailed breakdown

    Examples:
        >>> breakdown = get_score_breakdown(score)
        >>> print(breakdown['components'])
    """
    metadata = score.metadata

    return {
        'composite': score.composite,
        'tier': score.tier,
        'components': {
            'face_presence': metadata.get('face_presence_score', 0.0),
            'emotion': metadata.get('emotion_score_component', 0.0),
            'intimacy': metadata.get('intimacy_score_component', 0.0),
            'engagement': metadata.get('engagement_score_component', 0.0)
        },
        'raw_scores': {
            'intimacy': score.intimacy_score,
            'engagement': score.engagement_score
        },
        'face_info': {
            'count': score.face_count,
            'coverage': score.face_coverage
        }
    }


def compare_scores(score1: EmotionalScore,
                  score2: EmotionalScore) -> str:
    """
    Compare two EmotionalScores and determine which is better.

    Args:
        score1: First score
        score2: Second score

    Returns:
        'first', 'second', or 'tie'

    Examples:
        >>> result = compare_scores(score1, score2)
        >>> if result == 'first':
        ...     print("First photo is more emotionally significant")
    """
    if score1.composite > score2.composite:
        return 'first'
    elif score2.composite > score1.composite:
        return 'second'
    else:
        return 'tie'


def rank_scores(scores: List[EmotionalScore]) -> List[EmotionalScore]:
    """
    Rank EmotionalScores from highest to lowest composite score.

    Args:
        scores: List of EmotionalScore objects

    Returns:
        Sorted list (highest score first)

    Examples:
        >>> ranked = rank_scores(all_scores)
        >>> print(f"Best photo: {ranked[0].composite:.1f}")
    """
    return sorted(scores, key=lambda s: s.composite, reverse=True)


def filter_by_tier(scores: List[EmotionalScore],
                  tier: str) -> List[EmotionalScore]:
    """
    Filter scores by tier.

    Args:
        scores: List of EmotionalScore objects
        tier: Tier to filter by ('high', 'medium', or 'low')

    Returns:
        Filtered list

    Examples:
        >>> high_scores = filter_by_tier(all_scores, 'high')
        >>> print(f"Found {len(high_scores)} high-significance photos")
    """
    return [s for s in scores if s.tier == tier]


def get_statistics(scores: List[EmotionalScore]) -> Dict[str, Any]:
    """
    Calculate statistics across multiple scores.

    Args:
        scores: List of EmotionalScore objects

    Returns:
        Dictionary with statistics

    Examples:
        >>> stats = get_statistics(all_scores)
        >>> print(f"Average composite: {stats['avg_composite']:.1f}")
    """
    if not scores:
        return {
            'count': 0,
            'avg_composite': 0.0,
            'max_composite': 0.0,
            'min_composite': 0.0,
            'tier_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }

    composites = [s.composite for s in scores]

    return {
        'count': len(scores),
        'avg_composite': sum(composites) / len(composites),
        'max_composite': max(composites),
        'min_composite': min(composites),
        'tier_distribution': {
            'high': sum(1 for s in scores if s.tier == 'high'),
            'medium': sum(1 for s in scores if s.tier == 'medium'),
            'low': sum(1 for s in scores if s.tier == 'low')
        },
        'avg_face_count': sum(s.face_count for s in scores) / len(scores),
        'photos_with_faces': sum(1 for s in scores if s.face_count > 0)
    }
