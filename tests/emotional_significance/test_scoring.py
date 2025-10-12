"""
Tests for scoring components and composite scoring
"""

import pytest
from src.emotional_significance.data_classes import FaceDetection, EmotionalScore
from src.emotional_significance.scoring.components import (
    calculate_face_presence_score,
    calculate_emotion_score,
    calculate_intimacy_score_component,
    calculate_engagement_score_component
)
from src.emotional_significance.scoring.composite import (
    create_emotional_score,
    determine_tier,
    calculate_composite_simple,
    compare_scores,
    rank_scores,
    filter_by_tier,
    get_statistics
)
from src.emotional_significance.config import ScoringWeightsConfig, TierThresholdsConfig


class TestFacePresenceScore:
    """Tests for face presence scoring."""

    def test_no_faces(self):
        """Test scoring with no faces."""
        score = calculate_face_presence_score(0, 0.0)
        assert score == 0.0

    def test_single_face_low_coverage(self):
        """Test single face with low coverage."""
        score = calculate_face_presence_score(1, 0.1)
        assert 10.0 <= score <= 15.0

    def test_single_face_high_coverage(self):
        """Test single face with high coverage."""
        score = calculate_face_presence_score(1, 0.8)
        assert 15.0 <= score <= 20.0

    def test_couple(self):
        """Test two faces (couple)."""
        score = calculate_face_presence_score(2, 0.4)
        assert 20.0 <= score <= 25.0

    def test_small_group(self):
        """Test small group (3-5 faces)."""
        score = calculate_face_presence_score(4, 0.5)
        assert 25.0 <= score <= 28.0

    def test_large_group(self):
        """Test large group (6+ faces)."""
        score = calculate_face_presence_score(8, 0.6)
        assert 28.0 <= score <= 30.0

    def test_max_score_clamping(self):
        """Test that score doesn't exceed max."""
        score = calculate_face_presence_score(20, 1.0)
        assert score <= 30.0


class TestEmotionScore:
    """Tests for emotion scoring."""

    def test_no_faces(self):
        """Test emotion score with no faces."""
        faces = []
        score = calculate_emotion_score(faces)
        assert score == 0.0

    def test_no_smiles(self):
        """Test with faces but no smiles."""
        faces = [
            FaceDetection(
                bbox=(0, 0, 100, 100),
                confidence=0.9,
                center=(50, 50),
                size_ratio=0.1,
                smile_confidence=0.3  # Below threshold
            )
        ]
        score = calculate_emotion_score(faces)
        assert score == 5.0  # Minimal score for no smiles

    def test_single_smile(self):
        """Test with single smiling face."""
        faces = [
            FaceDetection(
                bbox=(0, 0, 100, 100),
                confidence=0.9,
                center=(50, 50),
                size_ratio=0.1,
                smile_confidence=0.8
            )
        ]
        score = calculate_emotion_score(faces)
        assert 30.0 < score <= 40.0

    def test_all_smiling(self):
        """Test with all faces smiling."""
        faces = [
            FaceDetection(
                bbox=(0, 0, 100, 100),
                confidence=0.9,
                center=(50, 50),
                size_ratio=0.1,
                smile_confidence=0.9
            )
            for _ in range(3)
        ]
        score = calculate_emotion_score(faces)
        assert 35.0 < score <= 40.0

    def test_partial_smiling(self):
        """Test with some faces smiling."""
        faces = [
            FaceDetection(
                bbox=(0, 0, 100, 100),
                confidence=0.9,
                center=(50, 50),
                size_ratio=0.1,
                smile_confidence=0.8
            ),
            FaceDetection(
                bbox=(100, 0, 100, 100),
                confidence=0.9,
                center=(150, 50),
                size_ratio=0.1,
                smile_confidence=0.3
            )
        ]
        score = calculate_emotion_score(faces)
        assert 15.0 < score < 30.0


class TestIntimacyScoreComponent:
    """Tests for intimacy score component."""

    def test_zero_intimacy(self):
        """Test zero intimacy."""
        score = calculate_intimacy_score_component(0.0)
        assert score == 0.0

    def test_medium_intimacy(self):
        """Test medium intimacy."""
        score = calculate_intimacy_score_component(50.0)
        assert score == 10.0  # 50% of 20

    def test_high_intimacy(self):
        """Test high intimacy."""
        score = calculate_intimacy_score_component(90.0)
        assert score == 18.0  # 90% of 20

    def test_max_intimacy(self):
        """Test maximum intimacy."""
        score = calculate_intimacy_score_component(100.0)
        assert score == 20.0


class TestEngagementScoreComponent:
    """Tests for engagement score component."""

    def test_zero_engagement(self):
        """Test zero engagement."""
        score = calculate_engagement_score_component(0.0)
        assert score == 0.0

    def test_medium_engagement(self):
        """Test medium engagement."""
        score = calculate_engagement_score_component(60.0)
        assert score == 6.0  # 60% of 10

    def test_high_engagement(self):
        """Test high engagement."""
        score = calculate_engagement_score_component(95.0)
        assert score == 9.5  # 95% of 10

    def test_max_engagement(self):
        """Test maximum engagement."""
        score = calculate_engagement_score_component(100.0)
        assert score == 10.0


class TestCompositScoring:
    """Tests for composite scoring."""

    def test_determine_tier_high(self):
        """Test high tier determination."""
        thresholds = TierThresholdsConfig()
        tier = determine_tier(85.0, thresholds)
        assert tier == 'high'

    def test_determine_tier_medium(self):
        """Test medium tier determination."""
        thresholds = TierThresholdsConfig()
        tier = determine_tier(55.0, thresholds)
        assert tier == 'medium'

    def test_determine_tier_low(self):
        """Test low tier determination."""
        thresholds = TierThresholdsConfig()
        tier = determine_tier(25.0, thresholds)
        assert tier == 'low'

    def test_calculate_composite_simple(self):
        """Test simple composite calculation."""
        composite = calculate_composite_simple(25.0, 35.0, 18.0, 9.0)
        assert composite == 87.0

    def test_composite_clamping(self):
        """Test composite score clamping."""
        # Test upper clamp
        composite = calculate_composite_simple(40.0, 50.0, 30.0, 20.0)
        assert composite == 100.0

        # Test lower clamp (shouldn't happen normally)
        composite = calculate_composite_simple(-5.0, 0.0, 0.0, 0.0)
        assert composite == 0.0

    def test_create_emotional_score(self):
        """Test creating EmotionalScore."""
        faces = [
            FaceDetection(
                bbox=(0, 0, 100, 100),
                confidence=0.9,
                center=(50, 50),
                size_ratio=0.1,
                smile_confidence=0.8
            )
        ]

        score = create_emotional_score(
            faces=faces,
            face_coverage=0.15,
            intimacy_raw_score=0.0,
            engagement_raw_score=90.0,
            weights=ScoringWeightsConfig(),
            thresholds=TierThresholdsConfig(),
            metadata={}
        )

        assert isinstance(score, EmotionalScore)
        assert score.face_count == 1
        assert 0.0 <= score.composite <= 100.0
        assert score.tier in ['high', 'medium', 'low']

    def test_compare_scores(self):
        """Test comparing two scores."""
        score1 = EmotionalScore(
            face_count=2, face_coverage=0.3, emotion_score=80.0,
            intimacy_score=85.0, engagement_score=90.0,
            composite=85.0, tier='high', metadata={}
        )
        score2 = EmotionalScore(
            face_count=1, face_coverage=0.2, emotion_score=50.0,
            intimacy_score=40.0, engagement_score=60.0,
            composite=55.0, tier='medium', metadata={}
        )

        result = compare_scores(score1, score2)
        assert result == 'first'

        result = compare_scores(score2, score1)
        assert result == 'second'

        result = compare_scores(score1, score1)
        assert result == 'tie'

    def test_rank_scores(self):
        """Test ranking scores."""
        scores = [
            EmotionalScore(
                face_count=1, face_coverage=0.2, emotion_score=50.0,
                intimacy_score=40.0, engagement_score=60.0,
                composite=55.0, tier='medium', metadata={}
            ),
            EmotionalScore(
                face_count=2, face_coverage=0.3, emotion_score=80.0,
                intimacy_score=85.0, engagement_score=90.0,
                composite=85.0, tier='high', metadata={}
            ),
            EmotionalScore(
                face_count=0, face_coverage=0.0, emotion_score=0.0,
                intimacy_score=0.0, engagement_score=0.0,
                composite=0.0, tier='low', metadata={}
            )
        ]

        ranked = rank_scores(scores)

        assert ranked[0].composite == 85.0
        assert ranked[1].composite == 55.0
        assert ranked[2].composite == 0.0

    def test_filter_by_tier(self):
        """Test filtering by tier."""
        scores = [
            EmotionalScore(
                face_count=1, face_coverage=0.2, emotion_score=50.0,
                intimacy_score=40.0, engagement_score=60.0,
                composite=55.0, tier='medium', metadata={}
            ),
            EmotionalScore(
                face_count=2, face_coverage=0.3, emotion_score=80.0,
                intimacy_score=85.0, engagement_score=90.0,
                composite=85.0, tier='high', metadata={}
            ),
            EmotionalScore(
                face_count=0, face_coverage=0.0, emotion_score=0.0,
                intimacy_score=0.0, engagement_score=0.0,
                composite=0.0, tier='low', metadata={}
            )
        ]

        high_scores = filter_by_tier(scores, 'high')
        assert len(high_scores) == 1
        assert high_scores[0].composite == 85.0

    def test_get_statistics(self):
        """Test getting statistics."""
        scores = [
            EmotionalScore(
                face_count=1, face_coverage=0.2, emotion_score=50.0,
                intimacy_score=40.0, engagement_score=60.0,
                composite=55.0, tier='medium', metadata={}
            ),
            EmotionalScore(
                face_count=2, face_coverage=0.3, emotion_score=80.0,
                intimacy_score=85.0, engagement_score=90.0,
                composite=85.0, tier='high', metadata={}
            )
        ]

        stats = get_statistics(scores)

        assert stats['count'] == 2
        assert stats['avg_composite'] == 70.0
        assert stats['max_composite'] == 85.0
        assert stats['min_composite'] == 55.0
        assert stats['tier_distribution']['high'] == 1
        assert stats['tier_distribution']['medium'] == 1
