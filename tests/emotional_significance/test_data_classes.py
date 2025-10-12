"""
Tests for data classes (FaceDetection, EmotionalScore)
"""

import pytest
from src.emotional_significance.data_classes import FaceDetection, EmotionalScore


class TestFaceDetection:
    """Tests for FaceDetection data class."""

    def test_face_detection_creation(self):
        """Test basic FaceDetection creation."""
        face = FaceDetection(
            bbox=(100, 100, 200, 250),
            confidence=0.95,
            center=(200, 225),
            size_ratio=0.15
        )

        assert face.bbox == (100, 100, 200, 250)
        assert face.confidence == 0.95
        assert face.center == (200, 225)
        assert face.size_ratio == 0.15
        assert face.smile_confidence is None

    def test_face_area_property(self):
        """Test face area calculation."""
        face = FaceDetection(
            bbox=(0, 0, 100, 200),
            confidence=0.9,
            center=(50, 100),
            size_ratio=0.1
        )

        assert face.area == 20000  # 100 * 200

    def test_is_smiling_property(self):
        """Test is_smiling property."""
        # No smile
        face1 = FaceDetection(
            bbox=(0, 0, 100, 100),
            confidence=0.9,
            center=(50, 50),
            size_ratio=0.1,
            smile_confidence=None
        )
        assert not face1.is_smiling

        # Weak smile
        face2 = FaceDetection(
            bbox=(0, 0, 100, 100),
            confidence=0.9,
            center=(50, 50),
            size_ratio=0.1,
            smile_confidence=0.3
        )
        assert not face2.is_smiling

        # Clear smile
        face3 = FaceDetection(
            bbox=(0, 0, 100, 100),
            confidence=0.9,
            center=(50, 50),
            size_ratio=0.1,
            smile_confidence=0.8
        )
        assert face3.is_smiling

    def test_face_dimensions(self):
        """Test width and height properties."""
        face = FaceDetection(
            bbox=(10, 20, 150, 200),
            confidence=0.9,
            center=(85, 120),
            size_ratio=0.1
        )

        assert face.width == 150
        assert face.height == 200

    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        face = FaceDetection(
            bbox=(0, 0, 100, 200),
            confidence=0.9,
            center=(50, 100),
            size_ratio=0.1
        )

        assert face.aspect_ratio == 0.5  # 100/200

    def test_to_dict(self):
        """Test conversion to dictionary."""
        face = FaceDetection(
            bbox=(10, 20, 30, 40),
            confidence=0.85,
            center=(25, 40),
            size_ratio=0.05,
            smile_confidence=0.7
        )

        face_dict = face.to_dict()

        assert face_dict['bbox'] == (10, 20, 30, 40)
        assert face_dict['confidence'] == 0.85
        assert face_dict['center'] == (25, 40)
        assert face_dict['size_ratio'] == 0.05
        assert face_dict['smile_confidence'] == 0.7
        assert face_dict['area'] == 1200
        assert face_dict['is_smiling'] is True


class TestEmotionalScore:
    """Tests for EmotionalScore data class."""

    def test_emotional_score_creation(self):
        """Test basic EmotionalScore creation."""
        score = EmotionalScore(
            face_count=2,
            face_coverage=0.35,
            emotion_score=85.0,
            intimacy_score=90.0,
            engagement_score=80.0,
            composite=88.0,
            tier='high',
            metadata={'test': 'data'}
        )

        assert score.face_count == 2
        assert score.face_coverage == 0.35
        assert score.emotion_score == 85.0
        assert score.intimacy_score == 90.0
        assert score.engagement_score == 80.0
        assert score.composite == 88.0
        assert score.tier == 'high'
        assert score.metadata == {'test': 'data'}

    def test_str_representation(self):
        """Test string representation."""
        score = EmotionalScore(
            face_count=2,
            face_coverage=0.35,
            emotion_score=85.0,
            intimacy_score=90.0,
            engagement_score=80.0,
            composite=88.0,
            tier='high',
            metadata={}
        )

        str_repr = str(score)
        assert 'EmotionalScore' in str_repr
        assert '88.0' in str_repr
        assert 'faces=2' in str_repr
        assert 'high' in str_repr

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = EmotionalScore(
            face_count=3,
            face_coverage=0.5,
            emotion_score=70.0,
            intimacy_score=60.0,
            engagement_score=85.0,
            composite=75.0,
            tier='high',
            metadata={'key': 'value'}
        )

        score_dict = score.to_dict()

        assert score_dict['face_count'] == 3
        assert score_dict['face_coverage'] == 0.5
        assert score_dict['emotion_score'] == 70.0
        assert score_dict['composite'] == 75.0
        assert score_dict['tier'] == 'high'
        assert score_dict['metadata'] == {'key': 'value'}

    def test_significance_properties(self):
        """Test significance level properties."""
        high_score = EmotionalScore(
            face_count=2, face_coverage=0.3, emotion_score=80.0,
            intimacy_score=85.0, engagement_score=90.0,
            composite=85.0, tier='high', metadata={}
        )
        assert high_score.is_high_significance
        assert not high_score.is_medium_significance
        assert not high_score.is_low_significance

        medium_score = EmotionalScore(
            face_count=1, face_coverage=0.2, emotion_score=50.0,
            intimacy_score=40.0, engagement_score=60.0,
            composite=55.0, tier='medium', metadata={}
        )
        assert not medium_score.is_high_significance
        assert medium_score.is_medium_significance
        assert not medium_score.is_low_significance

        low_score = EmotionalScore(
            face_count=0, face_coverage=0.0, emotion_score=0.0,
            intimacy_score=0.0, engagement_score=0.0,
            composite=0.0, tier='low', metadata={}
        )
        assert not low_score.is_high_significance
        assert not low_score.is_medium_significance
        assert low_score.is_low_significance

    def test_face_properties(self):
        """Test face-related properties."""
        # No faces
        score1 = EmotionalScore(
            face_count=0, face_coverage=0.0, emotion_score=0.0,
            intimacy_score=0.0, engagement_score=0.0,
            composite=0.0, tier='low', metadata={}
        )
        assert not score1.has_faces
        assert not score1.has_multiple_people

        # Single face
        score2 = EmotionalScore(
            face_count=1, face_coverage=0.2, emotion_score=30.0,
            intimacy_score=0.0, engagement_score=80.0,
            composite=45.0, tier='medium', metadata={}
        )
        assert score2.has_faces
        assert not score2.has_multiple_people

        # Multiple faces
        score3 = EmotionalScore(
            face_count=3, face_coverage=0.5, emotion_score=80.0,
            intimacy_score=70.0, engagement_score=85.0,
            composite=82.0, tier='high', metadata={}
        )
        assert score3.has_faces
        assert score3.has_multiple_people

    def test_positive_emotion_property(self):
        """Test positive emotion property."""
        low_emotion = EmotionalScore(
            face_count=1, face_coverage=0.2, emotion_score=30.0,
            intimacy_score=0.0, engagement_score=80.0,
            composite=45.0, tier='medium', metadata={}
        )
        assert not low_emotion.has_positive_emotion

        high_emotion = EmotionalScore(
            face_count=2, face_coverage=0.3, emotion_score=75.0,
            intimacy_score=80.0, engagement_score=85.0,
            composite=82.0, tier='high', metadata={}
        )
        assert high_emotion.has_positive_emotion
