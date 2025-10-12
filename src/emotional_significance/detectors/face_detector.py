"""
Face Detector - DNN-based Face Detection

This module implements face detection using OpenCV's DNN module with
a pre-trained ResNet-10 SSD model. This approach provides better accuracy
and fewer false positives compared to Haar Cascades.

Performance: ~10-15ms per image (1024px) on CPU
Accuracy: >95% on clear frontal faces, works with faces at various angles
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from ..data_classes import FaceDetection
from ..config import FaceDetectionConfig

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    DNN-based face detector using OpenCV's pre-trained ResNet-10 SSD model.

    This detector provides accurate face detection with configurable confidence
    thresholds and size filtering. It's optimized for speed while maintaining
    high accuracy.

    Attributes:
        config: Configuration for face detection parameters
        net: Loaded OpenCV DNN model

    Examples:
        >>> detector = FaceDetector()
        >>> faces = detector.detect_faces(image)
        >>> print(f"Found {len(faces)} faces")
        >>> for face in faces:
        ...     print(f"  Face at {face.center} with confidence {face.confidence:.2f}")
    """

    def __init__(self, config: Optional[FaceDetectionConfig] = None):
        """
        Initialize face detector.

        Args:
            config: Optional custom configuration. Uses default if not provided.

        Raises:
            FileNotFoundError: If model files are not found
            RuntimeError: If model fails to load
        """
        self.config = config if config is not None else FaceDetectionConfig()

        # Load DNN model
        try:
            model_dir = Path(__file__).parent.parent / 'models'
            prototxt, weights = self.config.get_model_paths(model_dir)

            if not prototxt.exists():
                raise FileNotFoundError(f"Model prototxt not found: {prototxt}")
            if not weights.exists():
                raise FileNotFoundError(f"Model weights not found: {weights}")

            self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
            logger.info("Face detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise RuntimeError(f"Failed to load face detection model: {e}")

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect all faces in image.

        Args:
            image: Input image as numpy array (RGB or BGR)
                   Shape: (height, width, 3) or (height, width)

        Returns:
            List of FaceDetection objects, sorted by size (largest first)

        Examples:
            >>> detector = FaceDetector()
            >>> faces = detector.detect_faces(image)
            >>> print(f"Found {len(faces)} faces")
            >>> largest_face = faces[0] if faces else None
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect_faces")
            return []

        h, w = image.shape[:2]
        image_area = h * w

        # Prepare blob for DNN (300x300 input size)
        try:
            blob = cv2.dnn.blobFromImage(
                image,
                scalefactor=self.config.dnn_scale_factor,
                size=self.config.dnn_input_size,
                mean=self.config.dnn_mean
            )
        except Exception as e:
            logger.error(f"Failed to create blob from image: {e}")
            return []

        # Run detection
        try:
            self.net.setInput(blob)
            detections = self.net.forward()
        except Exception as e:
            logger.error(f"Failed to run face detection: {e}")
            return []

        faces = []

        # Process detections
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            # Filter by confidence
            if confidence < self.config.confidence_threshold:
                continue

            # Get bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            # Ensure box is within image bounds
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face_w = x2 - x
            face_h = y2 - y

            # Skip invalid boxes
            if face_w <= 0 or face_h <= 0:
                continue

            # Calculate size ratio
            face_area = face_w * face_h
            size_ratio = face_area / image_area

            # Filter by size
            if size_ratio < self.config.min_face_size_ratio:
                continue
            if size_ratio > self.config.max_face_size_ratio:
                continue

            # Calculate center
            center_x = x + face_w // 2
            center_y = y + face_h // 2

            # Create FaceDetection object
            face = FaceDetection(
                bbox=(x, y, face_w, face_h),
                confidence=confidence,
                center=(center_x, center_y),
                size_ratio=float(size_ratio)
            )

            faces.append(face)

            # Stop if we've reached max faces
            if len(faces) >= self.config.max_faces:
                break

        # Sort by size (largest first)
        faces.sort(key=lambda f: f.size_ratio, reverse=True)

        logger.debug(f"Detected {len(faces)} faces in image")
        return faces

    def calculate_face_coverage(self,
                                faces: List[FaceDetection],
                                image_shape: Tuple[int, int]) -> float:
        """
        Calculate total percentage of image covered by faces.

        This is a simple approach that sums all face areas. For MVP, overlapping
        faces may be counted multiple times, but this is acceptable as it indicates
        a crowded, potentially significant photo.

        Args:
            faces: List of detected faces
            image_shape: Image dimensions (height, width)

        Returns:
            Coverage ratio (0.0-1.0)

        Examples:
            >>> coverage = detector.calculate_face_coverage(faces, image.shape[:2])
            >>> print(f"Faces cover {coverage*100:.1f}% of image")
        """
        if not faces:
            return 0.0

        h, w = image_shape
        image_area = h * w

        # Sum all face areas
        total_face_area = sum(f.area for f in faces)

        # Cap at 100% coverage
        coverage = min(1.0, total_face_area / image_area)

        return float(coverage)

    def get_largest_face(self, faces: List[FaceDetection]) -> Optional[FaceDetection]:
        """
        Get the largest detected face.

        Args:
            faces: List of detected faces

        Returns:
            Largest face or None if list is empty
        """
        return faces[0] if faces else None

    def filter_faces_by_confidence(self,
                                   faces: List[FaceDetection],
                                   min_confidence: float) -> List[FaceDetection]:
        """
        Filter faces by minimum confidence.

        Args:
            faces: List of detected faces
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            Filtered list of faces
        """
        return [f for f in faces if f.confidence >= min_confidence]

    def filter_faces_by_size(self,
                            faces: List[FaceDetection],
                            min_size_ratio: float) -> List[FaceDetection]:
        """
        Filter faces by minimum size ratio.

        Args:
            faces: List of detected faces
            min_size_ratio: Minimum size ratio (0.0-1.0)

        Returns:
            Filtered list of faces
        """
        return [f for f in faces if f.size_ratio >= min_size_ratio]


# Add helper method to FaceDetectionConfig for model paths
def _get_model_paths(config: FaceDetectionConfig, base_dir: Path) -> Tuple[Path, Path]:
    """Helper to get model file paths."""
    prototxt = base_dir / "deploy.prototxt"
    weights = base_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    return prototxt, weights


# Monkey-patch the method onto the config class
FaceDetectionConfig.get_model_paths = _get_model_paths
