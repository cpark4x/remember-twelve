"""
Emotional Significance Detector - Scoring Modules

This package contains modules for scoring different components of
emotional significance:
- components.py: Individual score components (face presence, emotion, intimacy, engagement)
- composite.py: Composite scoring and EmotionalScore creation
"""

from .components import (
    calculate_face_presence_score,
    calculate_emotion_score,
    calculate_intimacy_score_component,
    calculate_engagement_score_component
)
from .composite import create_emotional_score

__all__ = [
    'calculate_face_presence_score',
    'calculate_emotion_score',
    'calculate_intimacy_score_component',
    'calculate_engagement_score_component',
    'create_emotional_score'
]
