"""
Proximity Calculator - Intimacy/Closeness Analysis

This module calculates physical proximity between faces to assess intimacy
and emotional closeness in photos. Closer faces indicate embracing, couples,
or close friendships.

The algorithm:
1. Calculate pairwise distances between all face centers
2. Normalize by average face width (accounts for photo scale)
3. Score based on distance thresholds
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import math

from ..data_classes import FaceDetection
from ..config import ProximityConfig

logger = logging.getLogger(__name__)


class ProximityCalculator:
    """
    Calculator for physical proximity and intimacy between faces.

    Analyzes the distances between detected faces to assess emotional closeness.
    Closer faces receive higher intimacy scores.

    Attributes:
        config: Configuration for proximity thresholds

    Examples:
        >>> calculator = ProximityCalculator()
        >>> intimacy_score = calculator.calculate_intimacy_score(faces)
        >>> if intimacy_score > 80:
        ...     print("Very close/intimate photo!")
    """

    def __init__(self, config: Optional[ProximityConfig] = None):
        """
        Initialize proximity calculator.

        Args:
            config: Optional custom configuration. Uses default if not provided.
        """
        self.config = config if config is not None else ProximityConfig()
        logger.info("Proximity calculator initialized")

    def calculate_intimacy_score(self, faces: List[FaceDetection]) -> float:
        """
        Calculate overall intimacy score based on face proximities.

        Args:
            faces: List of detected faces

        Returns:
            Intimacy score (0-100)
                90-100: Very close (embracing, intimate)
                70-89: Close (friends, couples)
                50-69: Moderate distance (group photo)
                0-49: Distant (separate individuals)

        Examples:
            >>> calculator = ProximityCalculator()
            >>> score = calculator.calculate_intimacy_score(faces)
            >>> print(f"Intimacy: {score:.1f}")
        """
        if len(faces) == 0:
            return 0.0

        if len(faces) == 1:
            # Single person: no intimacy score
            return 0.0

        # Calculate all pairwise distances
        distances = self._calculate_pairwise_distances(faces)

        if not distances:
            return 0.0

        # Find minimum distance (closest pair)
        min_distance = min(distances)

        # Score based on minimum distance
        score = self._distance_to_score(min_distance)

        # Bonus for multiple close pairs
        close_pairs = sum(1 for d in distances if d < self.config.close_threshold)
        if close_pairs >= 2:
            # Multiple close pairs = group intimacy
            score = min(100.0, score + 10.0)

        return float(score)

    def _calculate_pairwise_distances(self,
                                     faces: List[FaceDetection]) -> List[float]:
        """
        Calculate normalized distances between all pairs of faces.

        Distance is normalized by average face width to account for photo scale.

        Args:
            faces: List of detected faces

        Returns:
            List of normalized distances
        """
        if len(faces) < 2:
            return []

        # Calculate average face width for normalization
        avg_face_width = sum(f.width for f in faces) / len(faces)

        if avg_face_width == 0:
            return []

        distances = []

        # Calculate distance for each pair
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                distance = self._calculate_normalized_distance(
                    faces[i],
                    faces[j],
                    avg_face_width
                )
                distances.append(distance)

        return distances

    def _calculate_normalized_distance(self,
                                       face1: FaceDetection,
                                       face2: FaceDetection,
                                       avg_face_width: float) -> float:
        """
        Calculate normalized Euclidean distance between two faces.

        Args:
            face1: First face
            face2: Second face
            avg_face_width: Average face width for normalization

        Returns:
            Normalized distance (unitless)
        """
        # Euclidean distance between centers
        dx = face1.center[0] - face2.center[0]
        dy = face1.center[1] - face2.center[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Normalize by average face width
        normalized = distance / avg_face_width if avg_face_width > 0 else float('inf')

        return normalized

    def _distance_to_score(self, distance: float) -> float:
        """
        Convert normalized distance to intimacy score.

        Args:
            distance: Normalized distance between faces

        Returns:
            Intimacy score (0-100)
        """
        if distance < self.config.very_close_threshold:
            # Very close: embracing, very intimate
            return self.config.very_close_score

        elif distance < self.config.close_threshold:
            # Close: interpolate between very_close and close
            ratio = (distance - self.config.very_close_threshold) / \
                    (self.config.close_threshold - self.config.very_close_threshold)
            return self.config.very_close_score - \
                   (self.config.very_close_score - self.config.close_score) * ratio

        elif distance < self.config.moderate_threshold:
            # Moderate: interpolate between close and moderate
            ratio = (distance - self.config.close_threshold) / \
                    (self.config.moderate_threshold - self.config.close_threshold)
            return self.config.close_score - \
                   (self.config.close_score - self.config.moderate_score) * ratio

        else:
            # Distant: interpolate between moderate and distant
            # Cap at 2x moderate_threshold for scoring
            max_distance = self.config.moderate_threshold * 2
            clamped_distance = min(distance, max_distance)

            ratio = (clamped_distance - self.config.moderate_threshold) / \
                    (max_distance - self.config.moderate_threshold)
            return self.config.moderate_score - \
                   (self.config.moderate_score - self.config.distant_score) * ratio

    def get_closest_pair(self,
                        faces: List[FaceDetection]) -> Optional[Tuple[FaceDetection, FaceDetection, float]]:
        """
        Find the closest pair of faces.

        Args:
            faces: List of detected faces

        Returns:
            Tuple of (face1, face2, distance) or None if < 2 faces

        Examples:
            >>> calculator = ProximityCalculator()
            >>> result = calculator.get_closest_pair(faces)
            >>> if result:
            ...     face1, face2, dist = result
            ...     print(f"Closest pair distance: {dist:.2f}")
        """
        if len(faces) < 2:
            return None

        avg_face_width = sum(f.width for f in faces) / len(faces)
        min_distance = float('inf')
        closest_pair = None

        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                distance = self._calculate_normalized_distance(
                    faces[i],
                    faces[j],
                    avg_face_width
                )

                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (faces[i], faces[j], distance)

        return closest_pair

    def get_proximity_analysis(self,
                              faces: List[FaceDetection]) -> Dict[str, any]:
        """
        Get detailed proximity analysis.

        Args:
            faces: List of detected faces

        Returns:
            Dictionary with analysis details:
                - intimacy_score: Overall score (0-100)
                - closest_distance: Minimum normalized distance
                - avg_distance: Average normalized distance
                - close_pairs: Number of close pairs
                - analysis: Text description

        Examples:
            >>> calculator = ProximityCalculator()
            >>> analysis = calculator.get_proximity_analysis(faces)
            >>> print(analysis['analysis'])
        """
        if len(faces) == 0:
            return {
                'intimacy_score': 0.0,
                'closest_distance': None,
                'avg_distance': None,
                'close_pairs': 0,
                'analysis': 'No faces detected'
            }

        if len(faces) == 1:
            return {
                'intimacy_score': 0.0,
                'closest_distance': None,
                'avg_distance': None,
                'close_pairs': 0,
                'analysis': 'Single person (no proximity to measure)'
            }

        # Calculate metrics
        distances = self._calculate_pairwise_distances(faces)
        min_distance = min(distances)
        avg_distance = sum(distances) / len(distances)
        intimacy_score = self.calculate_intimacy_score(faces)

        close_pairs = sum(1 for d in distances if d < self.config.close_threshold)

        # Generate analysis text
        if min_distance < self.config.very_close_threshold:
            analysis = "Very close/intimate (embracing or very near)"
        elif min_distance < self.config.close_threshold:
            analysis = "Close together (friends, couple)"
        elif min_distance < self.config.moderate_threshold:
            analysis = "Moderate distance (group photo)"
        else:
            analysis = "Distant (separate individuals)"

        return {
            'intimacy_score': float(intimacy_score),
            'closest_distance': float(min_distance),
            'avg_distance': float(avg_distance),
            'close_pairs': close_pairs,
            'analysis': analysis
        }
