"""
Tests for Twelve Curator data classes.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import json

from src.twelve_curator.data_classes import (
    PhotoCandidate,
    TwelveSelection,
    CurationConfig
)


class TestPhotoCandidate:
    """Tests for PhotoCandidate class."""

    def test_create_candidate(self):
        """Should create a photo candidate."""
        candidate = PhotoCandidate(
            photo_path=Path("test.jpg"),
            timestamp=datetime(2024, 6, 15, 14, 30),
            month=6,
            quality_score=75.5,
            emotional_score=60.2,
            combined_score=66.3,
            metadata={'has_faces': True}
        )

        assert candidate.photo_path == Path("test.jpg")
        assert candidate.month == 6
        assert candidate.quality_score == 75.5
        assert candidate.emotional_score == 60.2
        assert candidate.combined_score == 66.3
        assert candidate.metadata['has_faces'] is True

    def test_to_dict(self):
        """Should export to dictionary."""
        candidate = PhotoCandidate(
            photo_path=Path("test.jpg"),
            timestamp=datetime(2024, 6, 15, 14, 30),
            month=6,
            quality_score=75.5,
            emotional_score=60.2,
            combined_score=66.3,
            metadata={'has_faces': True}
        )

        data = candidate.to_dict()

        assert data['photo_path'] == 'test.jpg'
        assert data['month'] == 6
        assert data['quality_score'] == 75.5
        assert data['combined_score'] == 66.3
        assert '2024-06-15' in data['timestamp']

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            'photo_path': 'test.jpg',
            'timestamp': '2024-06-15T14:30:00',
            'month': 6,
            'quality_score': 75.5,
            'emotional_score': 60.2,
            'combined_score': 66.3,
            'metadata': {'has_faces': True}
        }

        candidate = PhotoCandidate.from_dict(data)

        assert candidate.photo_path == Path('test.jpg')
        assert candidate.month == 6
        assert candidate.quality_score == 75.5

    def test_str_representation(self):
        """Should have readable string representation."""
        candidate = PhotoCandidate(
            photo_path=Path("test.jpg"),
            timestamp=datetime(2024, 6, 15),
            month=6,
            quality_score=75.5,
            emotional_score=60.2,
            combined_score=66.3
        )

        s = str(candidate)
        assert 'test.jpg' in s
        assert 'Month 6' in s
        assert '66.3' in s


class TestTwelveSelection:
    """Tests for TwelveSelection class."""

    def test_create_selection(self):
        """Should create a selection."""
        photos = [
            PhotoCandidate(
                photo_path=Path(f"photo_{i}.jpg"),
                timestamp=datetime(2024, i, 1),
                month=i,
                quality_score=70.0 + i,
                emotional_score=60.0 + i,
                combined_score=64.0 + i
            )
            for i in range(1, 13)
        ]

        selection = TwelveSelection(
            year=2024,
            photos=photos,
            strategy='balanced',
            stats={'avg_combined': 70.0}
        )

        assert selection.year == 2024
        assert len(selection.photos) == 12
        assert selection.strategy == 'balanced'

    def test_validation_max_12_photos(self):
        """Should reject more than 12 photos."""
        photos = [
            PhotoCandidate(
                photo_path=Path(f"photo_{i}.jpg"),
                timestamp=datetime(2024, 1, i),
                month=1,
                quality_score=70.0,
                emotional_score=60.0,
                combined_score=64.0
            )
            for i in range(1, 14)  # 13 photos
        ]

        with pytest.raises(ValueError, match="cannot have more than 12"):
            TwelveSelection(
                year=2024,
                photos=photos,
                strategy='balanced',
                stats={}
            )

    def test_to_dict(self):
        """Should export to dictionary."""
        photos = [
            PhotoCandidate(
                photo_path=Path("photo_1.jpg"),
                timestamp=datetime(2024, 1, 1),
                month=1,
                quality_score=70.0,
                emotional_score=60.0,
                combined_score=64.0
            )
        ]

        selection = TwelveSelection(
            year=2024,
            photos=photos,
            strategy='balanced',
            stats={'avg_combined': 64.0}
        )

        data = selection.to_dict()

        assert data['year'] == 2024
        assert data['strategy'] == 'balanced'
        assert len(data['photos']) == 1
        assert 'stats' in data

    def test_save_and_load(self):
        """Should save and load from JSON."""
        photos = [
            PhotoCandidate(
                photo_path=Path("photo_1.jpg"),
                timestamp=datetime(2024, 1, 1),
                month=1,
                quality_score=70.0,
                emotional_score=60.0,
                combined_score=64.0
            )
        ]

        selection = TwelveSelection(
            year=2024,
            photos=photos,
            strategy='balanced',
            stats={'avg_combined': 64.0}
        )

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "selection.json"
            selection.save(output_path)

            assert output_path.exists()

            # Load
            loaded = TwelveSelection.load(output_path)

            assert loaded.year == 2024
            assert len(loaded.photos) == 1
            assert loaded.strategy == 'balanced'
            assert loaded.stats['avg_combined'] == 64.0

    def test_summary(self):
        """Should generate readable summary."""
        photos = [
            PhotoCandidate(
                photo_path=Path(f"photo_{i}.jpg"),
                timestamp=datetime(2024, i, 1),
                month=i,
                quality_score=70.0 + i,
                emotional_score=60.0 + i,
                combined_score=64.0 + i
            )
            for i in range(1, 13)
        ]

        selection = TwelveSelection(
            year=2024,
            photos=photos,
            strategy='balanced',
            stats={
                'total_candidates': 100,
                'avg_combined': 70.0,
                'avg_quality': 75.0,
                'avg_emotional': 65.0,
                'months_represented': 12,
                'photos_with_faces': 8
            }
        )

        summary = selection.summary()

        assert '2024' in summary
        assert 'balanced' in summary
        assert '12' in summary  # photo count
        assert 'photo_1.jpg' in summary


class TestCurationConfig:
    """Tests for CurationConfig class."""

    def test_default_config(self):
        """Should create default config."""
        config = CurationConfig()

        assert config.quality_weight == 0.4
        assert config.emotional_weight == 0.6
        assert config.strategy == "balanced"
        assert config.visual_diversity_threshold == 0.85
        assert config.enable_diversity_filter is True

    def test_balanced_preset(self):
        """Should create balanced preset."""
        config = CurationConfig.balanced()

        assert config.quality_weight == 0.4
        assert config.emotional_weight == 0.6
        assert config.strategy == "balanced"

    def test_aesthetic_first_preset(self):
        """Should create aesthetic-first preset."""
        config = CurationConfig.aesthetic_first()

        assert config.quality_weight == 0.7
        assert config.emotional_weight == 0.3
        assert config.strategy == "aesthetic_first"

    def test_people_first_preset(self):
        """Should create people-first preset."""
        config = CurationConfig.people_first()

        assert config.quality_weight == 0.3
        assert config.emotional_weight == 0.7
        assert config.strategy == "people_first"
        assert config.prefer_with_faces is True

    def test_top_heavy_preset(self):
        """Should create top-heavy preset."""
        config = CurationConfig.top_heavy()

        assert config.strategy == "top_heavy"
        assert config.enforce_monthly_distribution is False

    def test_validation_weight_range(self):
        """Should validate weight ranges."""
        with pytest.raises(ValueError, match="quality_weight must be between 0 and 1"):
            CurationConfig(quality_weight=1.5, emotional_weight=-0.5)

    def test_validation_weights_sum(self):
        """Should validate weights sum to 1."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            CurationConfig(quality_weight=0.5, emotional_weight=0.3)

    def test_validation_threshold_range(self):
        """Should validate threshold range."""
        with pytest.raises(ValueError, match="visual_diversity_threshold must be between"):
            CurationConfig(visual_diversity_threshold=1.5)

    def test_to_dict(self):
        """Should export to dictionary."""
        config = CurationConfig.balanced()
        data = config.to_dict()

        assert data['quality_weight'] == 0.4
        assert data['emotional_weight'] == 0.6
        assert data['strategy'] == 'balanced'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
