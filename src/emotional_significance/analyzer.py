"""
Emotional Significance Analyzer - Main Interface

This module provides the primary interface for analyzing emotional significance
in photos. It orchestrates face detection, smile detection, proximity analysis,
and engagement detection to produce comprehensive emotional scores.

Usage:
    >>> from emotional_significance.analyzer import EmotionalAnalyzer
    >>> analyzer = EmotionalAnalyzer()
    >>> score = analyzer.analyze_photo('path/to/photo.jpg')
    >>> print(f"Emotional significance: {score.composite:.1f} ({score.tier})")
"""

from typing import Union, Optional, List, Tuple, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import logging
import time

from .data_classes import FaceDetection, EmotionalScore
from .config import EmotionalConfig, get_default_config
from .detectors import FaceDetector, SmileDetector, ProximityCalculator, EngagementDetector
from .scoring import create_emotional_score

logger = logging.getLogger(__name__)


class EmotionalAnalyzer:
    """
    Main analyzer for emotional significance assessment.

    This class orchestrates all detection and scoring components to produce
    comprehensive emotional significance scores for photos.

    Attributes:
        config: Configuration object with all parameters
        face_detector: DNN-based face detector
        smile_detector: Haar Cascade smile detector
        proximity_calculator: Intimacy/closeness calculator
        engagement_detector: Face orientation/engagement detector

    Examples:
        >>> # Basic usage with default configuration
        >>> analyzer = EmotionalAnalyzer()
        >>> score = analyzer.analyze_photo('family_photo.jpg')
        >>> if score.tier == 'high':
        ...     print("Highly emotionally significant photo!")

        >>> # Custom configuration
        >>> from emotional_significance.config import create_custom_config
        >>> config = create_custom_config(
        ...     face_detection={'confidence_threshold': 0.7}
        ... )
        >>> analyzer = EmotionalAnalyzer(config=config)

        >>> # Analyze from numpy array
        >>> import cv2
        >>> image = cv2.imread('photo.jpg')
        >>> score = analyzer.analyze_image(image)
    """

    def __init__(self, config: Optional[EmotionalConfig] = None):
        """
        Initialize the EmotionalAnalyzer.

        Args:
            config: Optional custom configuration. Uses default if not provided.

        Raises:
            RuntimeError: If detector initialization fails
        """
        self.config = config if config is not None else get_default_config()
        self.config.validate()

        # Initialize detectors
        try:
            self.face_detector = FaceDetector(self.config.face_detection)
            self.smile_detector = SmileDetector(self.config.smile_detection)
            self.proximity_calculator = ProximityCalculator(self.config.proximity)
            self.engagement_detector = EngagementDetector(self.config.engagement)
            logger.info("EmotionalAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EmotionalAnalyzer: {e}")
            raise RuntimeError(f"Failed to initialize analyzer: {e}")

    def analyze_photo(self, photo_path: Union[str, Path]) -> EmotionalScore:
        """
        Analyze a photo file and return emotional significance score.

        This is the primary method for analyzing photos from file paths.
        It handles loading, preprocessing, detection, and scoring.

        Args:
            photo_path: Path to the photo file (string or Path object)

        Returns:
            EmotionalScore: Object containing all emotional significance metrics

        Raises:
            FileNotFoundError: If photo file doesn't exist
            ValueError: If file cannot be loaded as an image
            IOError: If file cannot be read

        Examples:
            >>> analyzer = EmotionalAnalyzer()
            >>> score = analyzer.analyze_photo('vacation/beach.jpg')
            >>> print(f"Faces: {score.face_count}")
            >>> print(f"Emotion: {score.emotion_score:.1f}")
            >>> print(f"Composite: {score.composite:.1f}")
            >>> print(f"Tier: {score.tier}")
        """
        # Convert to Path object for consistency
        path = Path(photo_path)

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"Photo file not found: {photo_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {photo_path}")

        # Load image
        try:
            image = self._load_image(path)
        except Exception as e:
            raise ValueError(f"Failed to load image from {photo_path}: {e}")

        # Analyze the image
        return self.analyze_image(image)

    def analyze_image(self, image: np.ndarray) -> EmotionalScore:
        """
        Analyze an image array and return emotional significance score.

        Use this method when you already have an image loaded as a numpy array
        (e.g., from video frames, API responses, or custom preprocessing).

        Args:
            image: Image as numpy array (RGB or BGR format)
                   Shape: (height, width, 3) or (height, width)

        Returns:
            EmotionalScore: Object containing all emotional significance metrics

        Raises:
            ValueError: If image is invalid
            TypeError: If image is not a numpy array

        Examples:
            >>> import cv2
            >>> analyzer = EmotionalAnalyzer()
            >>> image = cv2.imread('photo.jpg')
            >>> score = analyzer.analyze_image(image)

            >>> # From PIL Image
            >>> from PIL import Image
            >>> pil_img = Image.open('photo.jpg')
            >>> np_img = np.array(pil_img)
            >>> score = analyzer.analyze_image(np_img)
        """
        start_time = time.time()

        # Preprocess image (resize if needed)
        processed_image = self._preprocess_image(image)

        # Step 1: Detect faces
        faces = self.face_detector.detect_faces(processed_image)
        face_coverage = self.face_detector.calculate_face_coverage(
            faces,
            processed_image.shape[:2]
        )

        # Step 2: Detect smiles for each face
        if faces:
            for i, face in enumerate(faces):
                smile_conf = self.smile_detector.detect_smile(processed_image, face)
                # Update face with smile confidence
                faces[i] = FaceDetection(
                    bbox=face.bbox,
                    confidence=face.confidence,
                    center=face.center,
                    size_ratio=face.size_ratio,
                    smile_confidence=smile_conf,
                    landmarks=face.landmarks
                )

        # Step 3: Calculate intimacy (proximity between faces)
        intimacy_raw_score = self.proximity_calculator.calculate_intimacy_score(faces)

        # Step 4: Calculate engagement (faces looking at camera)
        engagement_raw_score = self.engagement_detector.calculate_engagement_score(faces)

        # Step 5: Create metadata
        metadata = {
            'num_smiling': sum(1 for f in faces if f.is_smiling),
            'avg_smile_confidence': (
                sum(f.smile_confidence for f in faces if f.smile_confidence is not None) / len(faces)
                if faces else 0.0
            ),
            'intimacy_analysis': self.proximity_calculator.get_proximity_analysis(faces) if len(faces) >= 2 else {},
            'engagement_analysis': self.engagement_detector.get_engagement_analysis(faces),
            'processing_time_ms': 0.0  # Will be updated below
        }

        # Step 6: Create composite emotional score
        score = create_emotional_score(
            faces=faces,
            face_coverage=face_coverage,
            intimacy_raw_score=intimacy_raw_score,
            engagement_raw_score=engagement_raw_score,
            weights=self.config.scoring_weights,
            thresholds=self.config.tier_thresholds,
            metadata=metadata
        )

        # Update processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        score.metadata['processing_time_ms'] = processing_time

        logger.debug(f"Analyzed image in {processing_time:.1f}ms: {score}")

        return score

    def analyze_batch(self,
                     photo_paths: List[Union[str, Path]]) -> List[Optional[EmotionalScore]]:
        """
        Analyze multiple photos and return their emotional significance scores.

        Note: This is a simple sequential implementation for Phase 1.
        For parallel processing of large batches, use analyze_batch_parallel()
        or EmotionalBatchProcessor directly.

        Args:
            photo_paths: List of paths to photo files

        Returns:
            List[Optional[EmotionalScore]]: Scores for each photo (same order)
                                            None if photo couldn't be analyzed

        Examples:
            >>> analyzer = EmotionalAnalyzer()
            >>> photos = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
            >>> scores = analyzer.analyze_batch(photos)
            >>> high_sig = [p for p, s in zip(photos, scores)
            ...             if s and s.tier == 'high']
            >>> print(f"Found {len(high_sig)} highly significant photos")
        """
        scores = []
        for photo_path in photo_paths:
            try:
                score = self.analyze_photo(photo_path)
                scores.append(score)
            except Exception as e:
                # Log error and continue with next photo
                logger.warning(f"Failed to analyze {photo_path}: {e}")
                scores.append(None)

        return scores

    def analyze_batch_parallel(
        self,
        photo_paths: List[Union[str, Path]],
        num_workers: int = 4,
        progress_callback: Optional[Any] = None
    ) -> List[Optional[EmotionalScore]]:
        """
        Analyze multiple photos in parallel using BatchProcessor.

        This method provides parallel processing with significantly better
        performance for large batches compared to analyze_batch().

        Args:
            photo_paths: List of paths to photo files
            num_workers: Number of parallel workers (defaults to 4)
            progress_callback: Optional callback(analyzed, total, failed) for progress

        Returns:
            List[Optional[EmotionalScore]]: Scores for each photo (same order)
                                            None if photo couldn't be analyzed

        Examples:
            >>> analyzer = EmotionalAnalyzer()
            >>> photos = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
            >>>
            >>> # Simple usage
            >>> scores = analyzer.analyze_batch_parallel(photos)
            >>>
            >>> # With progress tracking
            >>> def on_progress(analyzed, total, failed):
            ...     print(f"Progress: {analyzed}/{total}")
            >>> scores = analyzer.analyze_batch_parallel(
            ...     photos,
            ...     num_workers=4,
            ...     progress_callback=on_progress
            ... )
        """
        from .batch_processor import EmotionalBatchProcessor

        processor = EmotionalBatchProcessor(num_workers=num_workers, config=self.config)
        result = processor.process_batch(photo_paths, progress_callback=progress_callback)

        # Create a map of path -> score for successful analyses
        score_map = {path: score for path, score in result.scores}

        # Return scores in original order, None for failed photos
        scores = []
        for path in photo_paths:
            path_str = str(path)
            scores.append(score_map.get(path_str))

        return scores

    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load image from file path.

        Supports common image formats: JPEG, PNG, BMP, TIFF, etc.

        Args:
            path: Path to image file

        Returns:
            np.ndarray: Image array in BGR format (OpenCV convention)

        Raises:
            ValueError: If image cannot be loaded
        """
        # Try OpenCV first (faster for most formats)
        image = cv2.imread(str(path))

        if image is None:
            # Fallback to PIL for other formats
            try:
                pil_image = Image.open(path)
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                # Convert to numpy array
                image = np.array(pil_image)
                # Convert RGB to BGR for OpenCV compatibility
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"Failed to load image with PIL: {e}")

        return image

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for analysis.

        Applies optimizations like resizing to improve performance while
        maintaining detection accuracy.

        Args:
            image: Input image array

        Returns:
            np.ndarray: Preprocessed image array

        Notes:
            - Resizes to max dimension of 1024px (configurable)
            - Maintains aspect ratio
            - Results in ~10x faster processing with minimal accuracy loss
        """
        max_size = self.config.performance.max_image_size
        height, width = image.shape[:2]

        # Check if resizing is needed
        if max(height, width) > max_size:
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            # Resize using high-quality interpolation
            image = cv2.resize(image, (new_width, new_height),
                             interpolation=cv2.INTER_LANCZOS4)

        return image

    def get_config(self) -> EmotionalConfig:
        """
        Get current configuration.

        Returns:
            EmotionalConfig: Current configuration object
        """
        return self.config

    def update_config(self, config: EmotionalConfig) -> None:
        """
        Update analyzer configuration.

        Note: This will reinitialize all detectors with new config.

        Args:
            config: New configuration object

        Raises:
            ValueError: If configuration is invalid
        """
        config.validate()
        self.config = config

        # Reinitialize detectors with new config
        self.face_detector = FaceDetector(self.config.face_detection)
        self.smile_detector = SmileDetector(self.config.smile_detection)
        self.proximity_calculator = ProximityCalculator(self.config.proximity)
        self.engagement_detector = EngagementDetector(self.config.engagement)

        logger.info("Configuration updated and detectors reinitialized")


def analyze_photo_simple(photo_path: Union[str, Path]) -> dict:
    """
    Convenience function for quick photo analysis.

    Simple one-line function for analyzing a photo without creating
    an analyzer instance. Uses default configuration.

    Args:
        photo_path: Path to photo file

    Returns:
        dict: Dictionary with emotional significance scores

    Examples:
        >>> result = analyze_photo_simple('photo.jpg')
        >>> print(f"Significance: {result['composite']:.1f}")
        >>> print(f"Faces: {result['face_count']}")
    """
    analyzer = EmotionalAnalyzer()
    score = analyzer.analyze_photo(photo_path)
    return score.to_dict()


# Example usage
if __name__ == '__main__':
    import sys

    print("Emotional Significance Analyzer - Phase 1")
    print("=" * 50)

    if len(sys.argv) > 1:
        photo_path = sys.argv[1]
        print(f"\nAnalyzing: {photo_path}")

        try:
            analyzer = EmotionalAnalyzer()
            score = analyzer.analyze_photo(photo_path)

            print(f"\nResults:")
            print(f"  Faces:      {score.face_count}")
            print(f"  Coverage:   {score.face_coverage*100:.1f}%")
            print(f"  Emotion:    {score.emotion_score:.1f}")
            print(f"  Intimacy:   {score.intimacy_score:.1f}")
            print(f"  Engagement: {score.engagement_score:.1f}")
            print(f"  Composite:  {score.composite:.1f}/100")
            print(f"  Tier:       {score.tier.upper()}")

            print(f"\nRecommendation:")
            if score.tier == 'high':
                print("  High emotional significance - prioritize for curation")
            elif score.tier == 'medium':
                print("  Medium emotional significance - include if space allows")
            else:
                print("  Low emotional significance - may exclude")

            if 'processing_time_ms' in score.metadata:
                print(f"\nProcessing time: {score.metadata['processing_time_ms']:.1f}ms")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    else:
        print("\nUsage:")
        print("  python analyzer.py <photo_path>")
        print("\nExample:")
        print("  python analyzer.py family_photo.jpg")
