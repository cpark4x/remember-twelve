"""
Smile Detector - Haar Cascade-based Smile Detection

This module implements smile detection using OpenCV's Haar Cascade classifier.
While less accurate than deep learning approaches, it's fast and good enough
for MVP emotional significance detection.

Performance: ~2ms per face region
Accuracy: ~80% on clear smiles
"""

import cv2
import numpy as np
from typing import Optional
import logging

from ..data_classes import FaceDetection
from ..config import SmileDetectionConfig

logger = logging.getLogger(__name__)


class SmileDetector:
    """
    Haar Cascade-based smile detector.

    Detects smiles within face regions using OpenCV's pre-trained classifier.
    Returns confidence scores based on the number and size of smile detections.

    Attributes:
        config: Configuration for smile detection parameters
        smile_cascade: Loaded Haar Cascade classifier

    Examples:
        >>> detector = SmileDetector()
        >>> smile_conf = detector.detect_smile(image, face)
        >>> if smile_conf > 0.5:
        ...     print(f"Smiling! Confidence: {smile_conf:.2f}")
    """

    def __init__(self, config: Optional[SmileDetectionConfig] = None):
        """
        Initialize smile detector.

        Args:
            config: Optional custom configuration. Uses default if not provided.

        Raises:
            RuntimeError: If Haar Cascade fails to load
        """
        self.config = config if config is not None else SmileDetectionConfig()

        # Load Haar Cascade smile classifier
        try:
            smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
            self.smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

            if self.smile_cascade.empty():
                raise RuntimeError("Failed to load Haar Cascade smile classifier")

            logger.info("Smile detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize smile detector: {e}")
            raise RuntimeError(f"Failed to load smile detection model: {e}")

    def detect_smile(self,
                     image: np.ndarray,
                     face: FaceDetection) -> float:
        """
        Detect smile in face region.

        Analyzes the face region for smile patterns and returns a confidence
        score based on the number and size of detections.

        Args:
            image: Full image as numpy array
            face: FaceDetection object with bounding box

        Returns:
            Smile confidence (0.0-1.0)
                0.0 = no smile detected
                0.5-0.8 = subtle smile
                0.8-1.0 = clear/strong smile

        Examples:
            >>> detector = SmileDetector()
            >>> smile_conf = detector.detect_smile(image, face)
            >>> if smile_conf > 0.8:
            ...     print("Strong smile detected!")
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect_smile")
            return 0.0

        # Extract face region
        x, y, w, h = face.bbox

        # Ensure bounds are valid
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            logger.warning(f"Invalid face bbox: {face.bbox}")
            return 0.0

        # Clip to image bounds
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return 0.0

        face_region = image[y:y+h, x:x+w]

        if face_region.size == 0:
            return 0.0

        # Convert to grayscale
        if len(face_region.shape) == 3:
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_region

        # Calculate minimum smile size
        min_smile_w = int(w * self.config.min_smile_width_ratio)
        min_smile_h = int(h * self.config.min_smile_height_ratio)

        # Ensure minimum size is reasonable
        min_smile_w = max(10, min_smile_w)
        min_smile_h = max(5, min_smile_h)

        # Detect smiles
        try:
            smiles = self.smile_cascade.detectMultiScale(
                face_gray,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=(min_smile_w, min_smile_h)
            )
        except Exception as e:
            logger.debug(f"Smile detection failed: {e}")
            return 0.0

        # Calculate confidence based on detections
        confidence = self._calculate_smile_confidence(smiles, w, h)

        return confidence

    def _calculate_smile_confidence(self,
                                    smiles: np.ndarray,
                                    face_width: int,
                                    face_height: int) -> float:
        """
        Calculate smile confidence from detected smile regions.

        Multiple or larger smile detections indicate higher confidence.

        Args:
            smiles: Array of detected smile regions [(x, y, w, h), ...]
            face_width: Width of face region
            face_height: Height of face region

        Returns:
            Confidence score (0.0-1.0)
        """
        if len(smiles) == 0:
            return 0.0

        # Base confidence on number of detections
        num_smiles = len(smiles)

        if num_smiles == 1:
            # Single detection: moderate confidence
            base_confidence = 0.6
        elif num_smiles == 2:
            # Two detections: high confidence
            base_confidence = 0.8
        else:
            # Multiple detections: very high confidence
            base_confidence = 0.9

        # Adjust based on size of largest smile detection
        largest_smile = max(smiles, key=lambda s: s[2] * s[3])
        smile_area = largest_smile[2] * largest_smile[3]
        face_area = face_width * face_height
        size_ratio = smile_area / face_area if face_area > 0 else 0.0

        # Larger smiles (relative to face) increase confidence
        if size_ratio > 0.2:  # Large smile region
            size_bonus = 0.1
        elif size_ratio > 0.1:  # Medium smile region
            size_bonus = 0.05
        else:  # Small smile region
            size_bonus = 0.0

        # Calculate final confidence
        confidence = min(1.0, base_confidence + size_bonus)

        return float(confidence)

    def detect_smiles_batch(self,
                           image: np.ndarray,
                           faces: list[FaceDetection]) -> list[float]:
        """
        Detect smiles for multiple faces in a single image.

        Args:
            image: Full image
            faces: List of FaceDetection objects

        Returns:
            List of smile confidences (same order as faces)

        Examples:
            >>> detector = SmileDetector()
            >>> confidences = detector.detect_smiles_batch(image, faces)
            >>> smiling_count = sum(1 for c in confidences if c > 0.5)
        """
        return [self.detect_smile(image, face) for face in faces]

    def update_face_with_smile(self,
                              image: np.ndarray,
                              face: FaceDetection) -> FaceDetection:
        """
        Update a FaceDetection object with smile information.

        Args:
            image: Full image
            face: FaceDetection object to update

        Returns:
            Updated FaceDetection with smile_confidence set

        Examples:
            >>> detector = SmileDetector()
            >>> face = detector.update_face_with_smile(image, face)
            >>> print(f"Smile confidence: {face.smile_confidence}")
        """
        smile_conf = self.detect_smile(image, face)

        # Create new FaceDetection with updated smile confidence
        updated_face = FaceDetection(
            bbox=face.bbox,
            confidence=face.confidence,
            center=face.center,
            size_ratio=face.size_ratio,
            smile_confidence=smile_conf,
            landmarks=face.landmarks
        )

        return updated_face

    def count_smiling_faces(self,
                           smile_confidences: list[float],
                           threshold: Optional[float] = None) -> int:
        """
        Count how many faces are smiling.

        Args:
            smile_confidences: List of smile confidence scores
            threshold: Optional custom threshold (uses config default if not provided)

        Returns:
            Number of smiling faces
        """
        threshold = threshold if threshold is not None else self.config.subtle_smile_threshold
        return sum(1 for conf in smile_confidences if conf >= threshold)
