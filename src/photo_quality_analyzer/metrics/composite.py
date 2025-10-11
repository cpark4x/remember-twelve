"""
Composite Score Module

This module combines individual quality metrics (sharpness, exposure) into a
single composite quality score using weighted averaging.

Formula (MVP): Quality Score = (Sharpness × 0.6) + (Exposure × 0.4)

Rationale:
- Sharpness weighted higher (users care more about blur than slight exposure issues)
- Simple weighted average (no ML required)
- Easy to tune weights based on user feedback
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# Default weights for composite scoring
DEFAULT_WEIGHTS = {
    'sharpness': 0.6,
    'exposure': 0.4
}


@dataclass
class QualityScore:
    """
    Container for photo quality scores.

    Attributes:
        sharpness: Sharpness score (0-100)
        exposure: Exposure score (0-100)
        composite: Overall quality score (0-100)
        tier: Quality tier ('high', 'acceptable', 'low')
    """
    sharpness: float
    exposure: float
    composite: float
    tier: str

    def __str__(self) -> str:
        """String representation of quality score."""
        return (f"QualityScore(composite={self.composite:.1f}, "
                f"sharpness={self.sharpness:.1f}, "
                f"exposure={self.exposure:.1f}, "
                f"tier='{self.tier}')")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'sharpness': self.sharpness,
            'exposure': self.exposure,
            'composite': self.composite,
            'tier': self.tier
        }


def calculate_quality_score(sharpness: float,
                           exposure: float,
                           weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate composite quality score from individual metrics.

    Uses weighted average to combine sharpness and exposure scores.
    Default weights: 60% sharpness, 40% exposure.

    Args:
        sharpness: Sharpness score (0-100)
        exposure: Exposure score (0-100)
        weights: Optional custom weights dict with 'sharpness' and 'exposure' keys.
                Must sum to 1.0. Defaults to DEFAULT_WEIGHTS.

    Returns:
        float: Composite quality score (0-100)

    Raises:
        ValueError: If scores are out of range or weights don't sum to 1.0

    Examples:
        >>> # Using default weights (60% sharpness, 40% exposure)
        >>> score = calculate_quality_score(sharpness=80, exposure=60)
        >>> print(f"Quality: {score:.1f}")
        Quality: 72.0

        >>> # Using custom weights
        >>> custom_weights = {'sharpness': 0.7, 'exposure': 0.3}
        >>> score = calculate_quality_score(80, 60, weights=custom_weights)
        >>> print(f"Quality: {score:.1f}")
        Quality: 74.0

        >>> # Edge case: perfect scores
        >>> score = calculate_quality_score(100, 100)
        >>> assert score == 100.0
    """
    # Input validation
    if not (0 <= sharpness <= 100):
        raise ValueError(f"Sharpness score must be 0-100, got {sharpness}")

    if not (0 <= exposure <= 100):
        raise ValueError(f"Exposure score must be 0-100, got {exposure}")

    # Use default weights if not provided
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Validate weights
    if 'sharpness' not in weights or 'exposure' not in weights:
        raise ValueError("Weights must contain 'sharpness' and 'exposure' keys")

    weight_sum = weights['sharpness'] + weights['exposure']
    if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point error
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    # Calculate weighted average
    composite = (sharpness * weights['sharpness'] +
                exposure * weights['exposure'])

    # Ensure result is in valid range (handle floating point precision)
    composite = max(0.0, min(100.0, composite))

    return float(composite)


def get_quality_tier(composite_score: float) -> str:
    """
    Convert composite score to quality tier.

    Quality tiers guide curation decisions:
    - High (70-100): Prioritize for curation
    - Acceptable (50-69): Include if needed for diversity/significance
    - Low (0-49): Exclude from curation entirely

    Args:
        composite_score: Composite quality score (0-100)

    Returns:
        str: Quality tier ('high', 'acceptable', 'low')

    Examples:
        >>> get_quality_tier(85.0)
        'high'
        >>> get_quality_tier(55.0)
        'acceptable'
        >>> get_quality_tier(30.0)
        'low'
    """
    if composite_score >= 70:
        return 'high'
    elif composite_score >= 50:
        return 'acceptable'
    else:
        return 'low'


def create_quality_score(sharpness: float,
                        exposure: float,
                        weights: Optional[Dict[str, float]] = None) -> QualityScore:
    """
    Create a QualityScore object from individual metrics.

    Convenience function that calculates composite score and determines tier.

    Args:
        sharpness: Sharpness score (0-100)
        exposure: Exposure score (0-100)
        weights: Optional custom weights for composite calculation

    Returns:
        QualityScore: Complete quality score object

    Examples:
        >>> score = create_quality_score(sharpness=80, exposure=60)
        >>> print(score)
        QualityScore(composite=72.0, sharpness=80.0, exposure=60.0, tier='high')

        >>> # Check tier
        >>> if score.tier == 'high':
        ...     print("High quality photo - include in curation")
        High quality photo - include in curation
    """
    composite = calculate_quality_score(sharpness, exposure, weights)
    tier = get_quality_tier(composite)

    return QualityScore(
        sharpness=float(sharpness),
        exposure=float(exposure),
        composite=composite,
        tier=tier
    )


def calculate_threshold_distances(composite_score: float) -> Dict[str, float]:
    """
    Calculate distance to quality tier thresholds.

    Useful for understanding how close a photo is to moving between tiers.

    Args:
        composite_score: Composite quality score (0-100)

    Returns:
        dict: Distances to thresholds:
            - to_acceptable: Points needed to reach acceptable tier (50)
            - to_high: Points needed to reach high tier (70)
            - current_tier: Current quality tier

    Examples:
        >>> distances = calculate_threshold_distances(45.0)
        >>> print(distances)
        {
            'to_acceptable': 5.0,
            'to_high': 25.0,
            'current_tier': 'low'
        }

        >>> # Photo is 5 points away from acceptable tier
        >>> if distances['to_acceptable'] < 10:
        ...     print("Close to acceptable quality")
        Close to acceptable quality
    """
    current_tier = get_quality_tier(composite_score)

    # Calculate distances to thresholds
    to_acceptable = max(0.0, 50.0 - composite_score)
    to_high = max(0.0, 70.0 - composite_score)

    return {
        'to_acceptable': to_acceptable,
        'to_high': to_high,
        'current_tier': current_tier
    }


def compare_scores(score1: QualityScore, score2: QualityScore) -> Dict[str, any]:
    """
    Compare two quality scores.

    Useful for ranking photos or analyzing changes after edits.

    Args:
        score1: First quality score
        score2: Second quality score

    Returns:
        dict: Comparison results:
            - composite_diff: Difference in composite scores (score1 - score2)
            - sharpness_diff: Difference in sharpness scores
            - exposure_diff: Difference in exposure scores
            - better_score: Which score is better ('score1', 'score2', or 'equal')
            - tier_change: Tier transition if any

    Examples:
        >>> original = create_quality_score(60, 55)  # composite=58.0
        >>> edited = create_quality_score(75, 70)    # composite=73.0
        >>> comparison = compare_scores(original, edited)
        >>> print(f"Improvement: {comparison['composite_diff']:.1f} points")
        Improvement: -15.0 points
        >>> print(f"Tier change: {comparison['tier_change']}")
        Tier change: acceptable -> high
    """
    composite_diff = score1.composite - score2.composite
    sharpness_diff = score1.sharpness - score2.sharpness
    exposure_diff = score1.exposure - score2.exposure

    # Determine which is better
    if abs(composite_diff) < 0.1:  # Essentially equal
        better_score = 'equal'
    elif composite_diff > 0:
        better_score = 'score1'
    else:
        better_score = 'score2'

    # Check for tier change
    if score1.tier != score2.tier:
        tier_change = f"{score1.tier} -> {score2.tier}"
    else:
        tier_change = None

    return {
        'composite_diff': composite_diff,
        'sharpness_diff': sharpness_diff,
        'exposure_diff': exposure_diff,
        'better_score': better_score,
        'tier_change': tier_change
    }


def batch_calculate_scores(metrics_list: list) -> list:
    """
    Calculate quality scores for multiple photos efficiently.

    Args:
        metrics_list: List of (sharpness, exposure) tuples

    Returns:
        list: List of QualityScore objects

    Examples:
        >>> metrics = [(80, 75), (45, 50), (90, 85)]
        >>> scores = batch_calculate_scores(metrics)
        >>> high_quality = [s for s in scores if s.tier == 'high']
        >>> print(f"Found {len(high_quality)} high quality photos")
        Found 2 high quality photos
    """
    return [create_quality_score(sharpness, exposure)
            for sharpness, exposure in metrics_list]
