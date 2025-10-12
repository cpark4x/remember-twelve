"""
Emotional Significance Detector - Detection Modules

This package contains modules for detecting various emotional signals:
- FaceDetector: DNN-based face detection
- SmileDetector: Haar Cascade smile detection
- ProximityCalculator: Intimacy/closeness calculation
- EngagementDetector: Face orientation/engagement detection
"""

from .face_detector import FaceDetector
from .smile_detector import SmileDetector
from .proximity_calculator import ProximityCalculator
from .engagement_detector import EngagementDetector

__all__ = [
    'FaceDetector',
    'SmileDetector',
    'ProximityCalculator',
    'EngagementDetector'
]
