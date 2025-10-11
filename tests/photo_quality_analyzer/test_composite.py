"""
Unit tests for composite scoring module.

Tests cover:
- Weighted average calculation
- Quality tier assignment
- QualityScore dataclass
- Custom weights
- Edge cases
- Score comparison utilities
"""

import pytest
from src.photo_quality_analyzer.metrics.composite import (
    calculate_quality_score,
    get_quality_tier,
    create_quality_score,
    QualityScore,
    calculate_threshold_distances,
    compare_scores,
    batch_calculate_scores,
    DEFAULT_WEIGHTS
)


class TestQualityScoreCalculation:
    """Tests for calculate_quality_score function."""

    def test_default_weights(self):
        """Should use default weights (60% sharpness, 40% exposure)."""
        # (80 * 0.6) + (60 * 0.4) = 48 + 24 = 72
        score = calculate_quality_score(sharpness=80, exposure=60)
        assert abs(score - 72.0) < 0.1, f"Expected 72.0, got {score}"

    def test_perfect_scores(self):
        """Perfect scores should result in 100."""
        score = calculate_quality_score(sharpness=100, exposure=100)
        assert score == 100.0

    def test_zero_scores(self):
        """Zero scores should result in 0."""
        score = calculate_quality_score(sharpness=0, exposure=0)
        assert score == 0.0

    def test_custom_weights(self):
        """Should respect custom weights."""
        custom_weights = {'sharpness': 0.7, 'exposure': 0.3}
        # (80 * 0.7) + (60 * 0.3) = 56 + 18 = 74
        score = calculate_quality_score(80, 60, weights=custom_weights)
        assert abs(score - 74.0) < 0.1, f"Expected 74.0, got {score}"

    def test_equal_weights(self):
        """Equal weights should produce simple average."""
        equal_weights = {'sharpness': 0.5, 'exposure': 0.5}
        # (80 * 0.5) + (60 * 0.5) = 40 + 30 = 70
        score = calculate_quality_score(80, 60, weights=equal_weights)
        assert abs(score - 70.0) < 0.1, f"Expected 70.0, got {score}"

    def test_sharpness_dominant(self):
        """High sharpness weight should make sharpness more important."""
        sharp_weights = {'sharpness': 0.9, 'exposure': 0.1}
        # (80 * 0.9) + (20 * 0.1) = 72 + 2 = 74
        score = calculate_quality_score(80, 20, weights=sharp_weights)
        assert abs(score - 74.0) < 0.1, f"Expected 74.0, got {score}"

    def test_score_clamped_to_range(self):
        """Score should always be 0-100 even with edge cases."""
        # Test various combinations
        test_cases = [
            (0, 0),
            (100, 100),
            (50, 50),
            (100, 0),
            (0, 100),
        ]

        for sharpness, exposure in test_cases:
            score = calculate_quality_score(sharpness, exposure)
            assert 0 <= score <= 100, \
                f"Score must be 0-100, got {score} for ({sharpness}, {exposure})"


class TestQualityScoreValidation:
    """Tests for input validation."""

    def test_invalid_sharpness_low(self):
        """Sharpness < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Sharpness score must be 0-100"):
            calculate_quality_score(sharpness=-1, exposure=50)

    def test_invalid_sharpness_high(self):
        """Sharpness > 100 should raise ValueError."""
        with pytest.raises(ValueError, match="Sharpness score must be 0-100"):
            calculate_quality_score(sharpness=101, exposure=50)

    def test_invalid_exposure_low(self):
        """Exposure < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Exposure score must be 0-100"):
            calculate_quality_score(sharpness=50, exposure=-1)

    def test_invalid_exposure_high(self):
        """Exposure > 100 should raise ValueError."""
        with pytest.raises(ValueError, match="Exposure score must be 0-100"):
            calculate_quality_score(sharpness=50, exposure=101)

    def test_weights_must_sum_to_one(self):
        """Weights not summing to 1.0 should raise ValueError."""
        invalid_weights = {'sharpness': 0.5, 'exposure': 0.6}  # Sum = 1.1
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            calculate_quality_score(50, 50, weights=invalid_weights)

    def test_missing_weight_keys(self):
        """Missing weight keys should raise ValueError."""
        incomplete_weights = {'sharpness': 0.6}  # Missing 'exposure'
        with pytest.raises(ValueError, match="must contain 'sharpness' and 'exposure'"):
            calculate_quality_score(50, 50, weights=incomplete_weights)


class TestQualityTier:
    """Tests for get_quality_tier function."""

    def test_high_tier(self):
        """Scores 70-100 should be 'high'."""
        assert get_quality_tier(70) == 'high'
        assert get_quality_tier(85) == 'high'
        assert get_quality_tier(100) == 'high'

    def test_acceptable_tier(self):
        """Scores 50-69 should be 'acceptable'."""
        assert get_quality_tier(50) == 'acceptable'
        assert get_quality_tier(60) == 'acceptable'
        assert get_quality_tier(69) == 'acceptable'

    def test_low_tier(self):
        """Scores 0-49 should be 'low'."""
        assert get_quality_tier(0) == 'low'
        assert get_quality_tier(25) == 'low'
        assert get_quality_tier(49) == 'low'

    def test_boundary_cases(self):
        """Test exact boundary values."""
        assert get_quality_tier(49.9) == 'low'
        assert get_quality_tier(50.0) == 'acceptable'
        assert get_quality_tier(69.9) == 'acceptable'
        assert get_quality_tier(70.0) == 'high'


class TestQualityScoreDataclass:
    """Tests for QualityScore dataclass."""

    def test_create_quality_score(self):
        """Should create QualityScore with all fields."""
        score = create_quality_score(sharpness=80, exposure=60)

        assert score.sharpness == 80
        assert score.exposure == 60
        assert abs(score.composite - 72.0) < 0.1  # Default weights
        assert score.tier == 'high'

    def test_quality_score_to_dict(self):
        """Should convert to dictionary."""
        score = create_quality_score(sharpness=80, exposure=60)
        score_dict = score.to_dict()

        assert 'sharpness' in score_dict
        assert 'exposure' in score_dict
        assert 'composite' in score_dict
        assert 'tier' in score_dict

    def test_quality_score_string(self):
        """String representation should be readable."""
        score = create_quality_score(sharpness=80, exposure=60)
        score_str = str(score)

        assert 'QualityScore' in score_str
        assert 'composite' in score_str
        assert 'tier' in score_str


class TestThresholdDistances:
    """Tests for calculate_threshold_distances function."""

    def test_low_quality_distances(self):
        """Low quality score should show distances to both thresholds."""
        distances = calculate_threshold_distances(45.0)

        assert distances['current_tier'] == 'low'
        assert distances['to_acceptable'] == 5.0  # 50 - 45
        assert distances['to_high'] == 25.0  # 70 - 45

    def test_acceptable_quality_distances(self):
        """Acceptable quality should show distance to high threshold."""
        distances = calculate_threshold_distances(60.0)

        assert distances['current_tier'] == 'acceptable'
        assert distances['to_acceptable'] == 0.0  # Already acceptable
        assert distances['to_high'] == 10.0  # 70 - 60

    def test_high_quality_distances(self):
        """High quality should show zero distances."""
        distances = calculate_threshold_distances(80.0)

        assert distances['current_tier'] == 'high'
        assert distances['to_acceptable'] == 0.0
        assert distances['to_high'] == 0.0


class TestScoreComparison:
    """Tests for compare_scores function."""

    def test_compare_equal_scores(self):
        """Comparing equal scores should indicate equality."""
        score1 = create_quality_score(70, 65)
        score2 = create_quality_score(70, 65)
        comparison = compare_scores(score1, score2)

        assert comparison['better_score'] == 'equal'
        assert comparison['composite_diff'] == 0.0
        assert comparison['tier_change'] is None

    def test_compare_score1_better(self):
        """Score1 higher should be indicated."""
        score1 = create_quality_score(80, 75)  # ~78
        score2 = create_quality_score(60, 55)  # ~58
        comparison = compare_scores(score1, score2)

        assert comparison['better_score'] == 'score1'
        assert comparison['composite_diff'] > 0

    def test_compare_score2_better(self):
        """Score2 higher should be indicated."""
        score1 = create_quality_score(60, 55)  # ~58
        score2 = create_quality_score(80, 75)  # ~78
        comparison = compare_scores(score1, score2)

        assert comparison['better_score'] == 'score2'
        assert comparison['composite_diff'] < 0

    def test_tier_change_detection(self):
        """Should detect tier changes."""
        score1 = create_quality_score(60, 55)  # acceptable
        score2 = create_quality_score(80, 75)  # high
        comparison = compare_scores(score1, score2)

        assert comparison['tier_change'] == 'acceptable -> high'

    def test_no_tier_change(self):
        """Should indicate no tier change when tiers are same."""
        score1 = create_quality_score(72, 68)  # high
        score2 = create_quality_score(80, 75)  # high
        comparison = compare_scores(score1, score2)

        assert comparison['tier_change'] is None

    def test_metric_differences(self):
        """Should calculate individual metric differences."""
        score1 = create_quality_score(80, 60)
        score2 = create_quality_score(70, 50)
        comparison = compare_scores(score1, score2)

        assert comparison['sharpness_diff'] == 10.0  # 80 - 70
        assert comparison['exposure_diff'] == 10.0  # 60 - 50


class TestBatchCalculation:
    """Tests for batch_calculate_scores function."""

    def test_batch_empty(self):
        """Empty batch should return empty list."""
        scores = batch_calculate_scores([])
        assert len(scores) == 0

    def test_batch_single(self):
        """Single item batch should work."""
        scores = batch_calculate_scores([(80, 60)])
        assert len(scores) == 1
        assert scores[0].sharpness == 80
        assert scores[0].exposure == 60

    def test_batch_multiple(self):
        """Multiple items should all be processed."""
        metrics = [
            (80, 75),  # high
            (60, 55),  # acceptable
            (40, 45),  # low
        ]
        scores = batch_calculate_scores(metrics)

        assert len(scores) == 3
        assert scores[0].tier == 'high'
        assert scores[1].tier == 'acceptable'
        assert scores[2].tier == 'low'

    def test_batch_order_preserved(self):
        """Batch order should be preserved."""
        metrics = [(90, 85), (50, 50), (30, 35)]
        scores = batch_calculate_scores(metrics)

        assert scores[0].sharpness == 90
        assert scores[1].sharpness == 50
        assert scores[2].sharpness == 30


class TestDefaultWeights:
    """Tests for DEFAULT_WEIGHTS constant."""

    def test_default_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        total = DEFAULT_WEIGHTS['sharpness'] + DEFAULT_WEIGHTS['exposure']
        assert abs(total - 1.0) < 0.01, f"Weights should sum to 1.0, got {total}"

    def test_default_weights_values(self):
        """Default weights should be 60/40 split."""
        assert DEFAULT_WEIGHTS['sharpness'] == 0.6
        assert DEFAULT_WEIGHTS['exposure'] == 0.4
