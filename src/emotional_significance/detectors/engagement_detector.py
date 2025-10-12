"""
Engagement Detector - Face Orientation/Camera Engagement Analysis

This module detects whether faces are looking at the camera (frontal view)
or turned away (profile view). Frontal faces indicate engagement and
direct connection with the viewer.

The algorithm uses face aspect ratio (width/height) as a proxy for orientation:
- Frontal faces: aspect ratio ~0.75-1.25 (roughly square)
- Profile faces: aspect ratio <0.6 or >1.5 (elongated)
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from ..data_classes import FaceDetection
from ..config import EngagementConfig

logger = logging.getLogger(__name__)


class EngagementDetector:
    """
    Detector for face orientation and camera engagement.

    Analyzes face aspect ratios to determine if faces are looking at the
    camera (frontal) or turned away (profile). Frontal faces receive higher
    engagement scores.

    Attributes:
        config: Configuration for engagement thresholds

    Examples:
        >>> detector = EngagementDetector()
        >>> engagement_score = detector.calculate_engagement_score(faces)
        >>> if engagement_score > 80:
        ...     print("Highly engaged with camera!")
    """

    def __init__(self, config: Optional[EngagementConfig] = None):
        """
        Initialize engagement detector.

        Args:
            config: Optional custom configuration. Uses default if not provided.
        """
        self.config = config if config is not None else EngagementConfig()
        logger.info("Engagement detector initialized")

    def calculate_engagement_score(self, faces: List[FaceDetection]) -> float:
        """
        Calculate overall engagement score for all faces.

        Args:
            faces: List of detected faces

        Returns:
            Engagement score (0-100)
                90-100: All faces frontal (fully engaged)
                60-89: Most faces frontal (mostly engaged)
                30-59: Mixed frontal/profile
                0-29: Mostly profile (not engaged)

        Examples:
            >>> detector = EngagementDetector()
            >>> score = detector.calculate_engagement_score(faces)
            >>> print(f"Engagement: {score:.1f}")
        """
        if len(faces) == 0:
            return 0.0

        # Calculate engagement for each face
        face_scores = [self._calculate_face_engagement(face) for face in faces]

        # Average engagement across all faces
        avg_score = sum(face_scores) / len(face_scores)

        return float(avg_score)

    def _calculate_face_engagement(self, face: FaceDetection) -> float:
        """
        Calculate engagement score for a single face.

        Uses aspect ratio to estimate face orientation.

        Args:
            face: FaceDetection object

        Returns:
            Engagement score (0-100)
        """
        aspect_ratio = face.aspect_ratio

        # Check if frontal
        if self.config.frontal_ratio_min <= aspect_ratio <= self.config.frontal_ratio_max:
            # Frontal face: high engagement
            return self.config.frontal_score

        # Check if profile (significantly skewed aspect ratio)
        elif aspect_ratio < self.config.profile_ratio_threshold or \
             aspect_ratio > (1 / self.config.profile_ratio_threshold):
            # Profile face: low engagement
            return self.config.profile_score

        else:
            # Partial profile: medium engagement
            return self.config.partial_score

    def classify_face_orientation(self, face: FaceDetection) -> str:
        """
        Classify face orientation as frontal, partial, or profile.

        Args:
            face: FaceDetection object

        Returns:
            Classification: 'frontal', 'partial', or 'profile'

        Examples:
            >>> detector = EngagementDetector()
            >>> orientation = detector.classify_face_orientation(face)
            >>> print(f"Face is {orientation}")
        """
        aspect_ratio = face.aspect_ratio

        if self.config.frontal_ratio_min <= aspect_ratio <= self.config.frontal_ratio_max:
            return 'frontal'
        elif aspect_ratio < self.config.profile_ratio_threshold or \
             aspect_ratio > (1 / self.config.profile_ratio_threshold):
            return 'profile'
        else:
            return 'partial'

    def count_frontal_faces(self, faces: List[FaceDetection]) -> int:
        """
        Count how many faces are frontal (looking at camera).

        Args:
            faces: List of detected faces

        Returns:
            Number of frontal faces

        Examples:
            >>> detector = EngagementDetector()
            >>> frontal_count = detector.count_frontal_faces(faces)
            >>> print(f"{frontal_count} out of {len(faces)} faces are frontal")
        """
        return sum(1 for face in faces
                  if self.classify_face_orientation(face) == 'frontal')

    def get_engagement_breakdown(self,
                                faces: List[FaceDetection]) -> Dict[str, int]:
        """
        Get breakdown of face orientations.

        Args:
            faces: List of detected faces

        Returns:
            Dictionary with counts:
                - frontal: Number of frontal faces
                - partial: Number of partially turned faces
                - profile: Number of profile faces

        Examples:
            >>> detector = EngagementDetector()
            >>> breakdown = detector.get_engagement_breakdown(faces)
            >>> print(f"Frontal: {breakdown['frontal']}, "
            ...       f"Profile: {breakdown['profile']}")
        """
        orientations = [self.classify_face_orientation(face) for face in faces]

        return {
            'frontal': orientations.count('frontal'),
            'partial': orientations.count('partial'),
            'profile': orientations.count('profile')
        }

    def get_engagement_analysis(self,
                               faces: List[FaceDetection]) -> Dict[str, any]:
        """
        Get detailed engagement analysis.

        Args:
            faces: List of detected faces

        Returns:
            Dictionary with analysis details:
                - engagement_score: Overall score (0-100)
                - frontal_count: Number of frontal faces
                - profile_count: Number of profile faces
                - frontal_ratio: Percentage of frontal faces
                - analysis: Text description

        Examples:
            >>> detector = EngagementDetector()
            >>> analysis = detector.get_engagement_analysis(faces)
            >>> print(analysis['analysis'])
        """
        if len(faces) == 0:
            return {
                'engagement_score': 0.0,
                'frontal_count': 0,
                'profile_count': 0,
                'partial_count': 0,
                'frontal_ratio': 0.0,
                'analysis': 'No faces detected'
            }

        # Calculate metrics
        engagement_score = self.calculate_engagement_score(faces)
        breakdown = self.get_engagement_breakdown(faces)

        frontal_count = breakdown['frontal']
        partial_count = breakdown['partial']
        profile_count = breakdown['profile']

        frontal_ratio = frontal_count / len(faces) if len(faces) > 0 else 0.0

        # Generate analysis text
        if frontal_ratio >= 0.8:
            analysis = "High engagement (most faces looking at camera)"
        elif frontal_ratio >= 0.5:
            analysis = "Moderate engagement (about half faces looking at camera)"
        elif frontal_ratio >= 0.2:
            analysis = "Low engagement (few faces looking at camera)"
        else:
            analysis = "Minimal engagement (mostly profile views)"

        return {
            'engagement_score': float(engagement_score),
            'frontal_count': frontal_count,
            'partial_count': partial_count,
            'profile_count': profile_count,
            'frontal_ratio': float(frontal_ratio),
            'analysis': analysis
        }

    def is_group_engaged(self,
                        faces: List[FaceDetection],
                        threshold: float = 0.6) -> bool:
        """
        Check if group is engaged with camera.

        Args:
            faces: List of detected faces
            threshold: Minimum ratio of frontal faces (0.0-1.0)

        Returns:
            True if >= threshold of faces are frontal

        Examples:
            >>> detector = EngagementDetector()
            >>> if detector.is_group_engaged(faces):
            ...     print("Group is engaged with camera!")
        """
        if len(faces) == 0:
            return False

        frontal_count = self.count_frontal_faces(faces)
        frontal_ratio = frontal_count / len(faces)

        return frontal_ratio >= threshold
