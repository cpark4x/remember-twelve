"""
Tests for detector modules
"""

import pytest
import numpy as np
import cv2
from src.emotional_significance.data_classes import FaceDetection
from src.emotional_significance.detectors import (
    ProximityCalculator,
    EngagementDetector
)


class TestProximityCalculator:
    """Tests for ProximityCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return ProximityCalculator()

    def test_no_faces(self, calculator):
        """Test with no faces."""
        score = calculator.calculate_intimacy_score([])
        assert score == 0.0

    def test_single_face(self, calculator):
        """Test with single face."""
        face = FaceDetection(
            bbox=(100, 100, 200, 250),
            confidence=0.9,
            center=(200, 225),
            size_ratio=0.15
        )
        score = calculator.calculate_intimacy_score([face])
        assert score == 0.0  # Single face = no intimacy

    def test_two_faces_close(self, calculator):
        """Test two faces close together."""
        face1 = FaceDetection(
            bbox=(100, 100, 100, 100),
            confidence=0.9,
            center=(150, 150),
            size_ratio=0.1
        )
        face2 = FaceDetection(
            bbox=(250, 100, 100, 100),
            confidence=0.9,
            center=(300, 150),
            size_ratio=0.1
        )
        score = calculator.calculate_intimacy_score([face1, face2])
        assert 0.0 < score <= 100.0

    def test_two_faces_distant(self, calculator):
        """Test two faces far apart."""
        face1 = FaceDetection(
            bbox=(10, 100, 100, 100),
            confidence=0.9,
            center=(60, 150),
            size_ratio=0.1
        )
        face2 = FaceDetection(
            bbox=(900, 100, 100, 100),
            confidence=0.9,
            center=(950, 150),
            size_ratio=0.1
        )
        score = calculator.calculate_intimacy_score([face1, face2])
        assert 0.0 <= score < 50.0  # Distant faces

    def test_get_closest_pair(self, calculator):
        """Test finding closest pair."""
        face1 = FaceDetection(
            bbox=(100, 100, 100, 100),
            confidence=0.9,
            center=(150, 150),
            size_ratio=0.1
        )
        face2 = FaceDetection(
            bbox=(250, 100, 100, 100),
            confidence=0.9,
            center=(300, 150),
            size_ratio=0.1
        )

        result = calculator.get_closest_pair([face1, face2])

        assert result is not None
        assert len(result) == 3
        assert result[0] == face1
        assert result[1] == face2
        assert result[2] > 0.0  # Distance

    def test_get_closest_pair_no_faces(self, calculator):
        """Test closest pair with < 2 faces."""
        result = calculator.get_closest_pair([])
        assert result is None

        face = FaceDetection(
            bbox=(100, 100, 100, 100),
            confidence=0.9,
            center=(150, 150),
            size_ratio=0.1
        )
        result = calculator.get_closest_pair([face])
        assert result is None

    def test_get_proximity_analysis(self, calculator):
        """Test proximity analysis."""
        face1 = FaceDetection(
            bbox=(100, 100, 100, 100),
            confidence=0.9,
            center=(150, 150),
            size_ratio=0.1
        )
        face2 = FaceDetection(
            bbox=(250, 100, 100, 100),
            confidence=0.9,
            center=(300, 150),
            size_ratio=0.1
        )

        analysis = calculator.get_proximity_analysis([face1, face2])

        assert 'intimacy_score' in analysis
        assert 'closest_distance' in analysis
        assert 'avg_distance' in analysis
        assert 'close_pairs' in analysis
        assert 'analysis' in analysis


class TestEngagementDetector:
    """Tests for EngagementDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return EngagementDetector()

    def test_no_faces(self, detector):
        """Test with no faces."""
        score = detector.calculate_engagement_score([])
        assert score == 0.0

    def test_frontal_face(self, detector):
        """Test frontal face (aspect ratio ~1.0)."""
        face = FaceDetection(
            bbox=(100, 100, 100, 100),  # Square face
            confidence=0.9,
            center=(150, 150),
            size_ratio=0.1
        )
        score = detector.calculate_engagement_score([face])
        assert score > 80.0  # High engagement for frontal

    def test_profile_face(self, detector):
        """Test profile face (elongated aspect ratio)."""
        face = FaceDetection(
            bbox=(100, 100, 50, 150),  # Tall narrow face
            confidence=0.9,
            center=(125, 175),
            size_ratio=0.1
        )
        score = detector.calculate_engagement_score([face])
        # Profile should score lower
        assert 0.0 <= score < 80.0

    def test_classify_face_orientation(self, detector):
        """Test face orientation classification."""
        # Frontal
        frontal_face = FaceDetection(
            bbox=(100, 100, 100, 100),
            confidence=0.9,
            center=(150, 150),
            size_ratio=0.1
        )
        assert detector.classify_face_orientation(frontal_face) == 'frontal'

        # Profile
        profile_face = FaceDetection(
            bbox=(100, 100, 50, 150),
            confidence=0.9,
            center=(125, 175),
            size_ratio=0.1
        )
        orientation = detector.classify_face_orientation(profile_face)
        assert orientation in ['profile', 'partial']

    def test_count_frontal_faces(self, detector):
        """Test counting frontal faces."""
        faces = [
            FaceDetection(
                bbox=(100, 100, 100, 100),
                confidence=0.9,
                center=(150, 150),
                size_ratio=0.1
            ),
            FaceDetection(
                bbox=(200, 100, 50, 150),
                confidence=0.9,
                center=(225, 175),
                size_ratio=0.1
            )
        ]

        count = detector.count_frontal_faces(faces)
        assert count >= 0

    def test_get_engagement_breakdown(self, detector):
        """Test engagement breakdown."""
        faces = [
            FaceDetection(
                bbox=(100, 100, 100, 100),
                confidence=0.9,
                center=(150, 150),
                size_ratio=0.1
            ),
            FaceDetection(
                bbox=(200, 100, 100, 100),
                confidence=0.9,
                center=(250, 150),
                size_ratio=0.1
            )
        ]

        breakdown = detector.get_engagement_breakdown(faces)

        assert 'frontal' in breakdown
        assert 'partial' in breakdown
        assert 'profile' in breakdown
        assert breakdown['frontal'] + breakdown['partial'] + breakdown['profile'] == len(faces)

    def test_get_engagement_analysis(self, detector):
        """Test engagement analysis."""
        faces = [
            FaceDetection(
                bbox=(100, 100, 100, 100),
                confidence=0.9,
                center=(150, 150),
                size_ratio=0.1
            )
        ]

        analysis = detector.get_engagement_analysis(faces)

        assert 'engagement_score' in analysis
        assert 'frontal_count' in analysis
        assert 'profile_count' in analysis
        assert 'frontal_ratio' in analysis
        assert 'analysis' in analysis

    def test_is_group_engaged(self, detector):
        """Test group engagement check."""
        # All frontal
        frontal_faces = [
            FaceDetection(
                bbox=(i * 100, 100, 100, 100),
                confidence=0.9,
                center=(i * 100 + 50, 150),
                size_ratio=0.1
            )
            for i in range(3)
        ]

        result = detector.is_group_engaged(frontal_faces)
        assert isinstance(result, bool)
