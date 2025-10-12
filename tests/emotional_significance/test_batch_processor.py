"""
Tests for EmotionalBatchProcessor

Covers parallel processing, progress tracking, error handling,
chunked processing, and path validation.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

from src.emotional_significance.batch_processor import (
    EmotionalBatchProcessor,
    BatchResult,
    _process_single_photo
)
from src.emotional_significance.config import EmotionalConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for test photos."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_photos(temp_dir):
    """Create a set of test photos with faces."""
    photos = []

    # Create 10 test photos
    for i in range(10):
        path = Path(temp_dir) / f"photo_{i}.jpg"

        # Create simple test image
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        Image.fromarray(image).save(path)
        photos.append(str(path))

    return photos


@pytest.fixture
def processor():
    """Create batch processor instance."""
    return EmotionalBatchProcessor(num_workers=2)


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_batch_result_creation(self):
        """Test creating a BatchResult."""
        result = BatchResult(
            total_photos=100,
            successful=95,
            failed=5,
            scores=[],
            errors=[]
        )

        assert result.total_photos == 100
        assert result.successful == 95
        assert result.failed == 5

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = BatchResult(
            total_photos=100,
            successful=95,
            failed=5,
            scores=[],
            errors=[]
        )

        assert result.success_rate == 95.0

    def test_success_rate_zero_photos(self):
        """Test success rate with zero photos."""
        result = BatchResult(
            total_photos=0,
            successful=0,
            failed=0,
            scores=[],
            errors=[]
        )

        assert result.success_rate == 0.0

    def test_success_rate_all_failed(self):
        """Test success rate when all photos failed."""
        result = BatchResult(
            total_photos=10,
            successful=0,
            failed=10,
            scores=[],
            errors=[]
        )

        assert result.success_rate == 0.0


class TestProcessSinglePhoto:
    """Test the worker function for single photo processing."""

    def test_process_single_photo_success(self, test_photos):
        """Test successful photo processing."""
        photo_path = test_photos[0]
        config_dict = EmotionalConfig().to_dict()

        path, score, error = _process_single_photo(photo_path, config_dict)

        assert path == photo_path
        assert score is not None
        assert error is None
        assert 0 <= score.composite <= 100

    def test_process_single_photo_nonexistent(self):
        """Test processing non-existent photo."""
        photo_path = "/nonexistent/photo.jpg"
        config_dict = EmotionalConfig().to_dict()

        path, score, error = _process_single_photo(photo_path, config_dict)

        assert path == photo_path
        assert score is None
        assert error is not None
        assert "FileNotFoundError" in error or "No such file" in error

    def test_process_single_photo_corrupted(self, temp_dir):
        """Test processing corrupted photo file."""
        # Create a corrupted file
        corrupted_path = Path(temp_dir) / "corrupted.jpg"
        with open(corrupted_path, 'w') as f:
            f.write("This is not a valid image file")

        config_dict = EmotionalConfig().to_dict()

        path, score, error = _process_single_photo(str(corrupted_path), config_dict)

        assert path == str(corrupted_path)
        assert score is None
        assert error is not None


class TestEmotionalBatchProcessor:
    """Test EmotionalBatchProcessor class."""

    def test_initialization_default(self):
        """Test processor initialization with defaults."""
        processor = EmotionalBatchProcessor()

        assert processor.config is not None
        assert processor.num_workers == 4

    def test_initialization_custom(self):
        """Test processor initialization with custom values."""
        config = EmotionalConfig()
        processor = EmotionalBatchProcessor(
            num_workers=8,
            config=config
        )

        assert processor.config == config
        assert processor.num_workers == 8

    def test_process_batch_empty(self, processor):
        """Test processing empty batch."""
        result = processor.process_batch([])

        assert result.total_photos == 0
        assert result.successful == 0
        assert result.failed == 0
        assert len(result.scores) == 0
        assert len(result.errors) == 0

    def test_process_batch_single_photo(self, processor, test_photos):
        """Test processing single photo."""
        result = processor.process_batch([test_photos[0]])

        assert result.total_photos == 1
        assert result.successful == 1
        assert result.failed == 0
        assert len(result.scores) == 1
        assert len(result.errors) == 0

    def test_process_batch_multiple_photos(self, processor, test_photos):
        """Test processing multiple photos."""
        result = processor.process_batch(test_photos)

        assert result.total_photos == 10
        assert result.successful == 10
        assert result.failed == 0
        assert len(result.scores) == 10
        assert len(result.errors) == 0

        # Verify all photos were processed
        processed_paths = {path for path, _ in result.scores}
        assert len(processed_paths) == 10

    def test_process_batch_with_failures(self, processor, test_photos):
        """Test processing batch with some failures."""
        # Add non-existent photo
        mixed_paths = test_photos[:5] + ["/nonexistent/photo.jpg"]

        result = processor.process_batch(mixed_paths)

        assert result.total_photos == 6
        assert result.successful == 5
        assert result.failed == 1
        assert len(result.scores) == 5
        assert len(result.errors) == 1

    def test_process_batch_all_failures(self, processor):
        """Test processing batch where all photos fail."""
        bad_paths = [f"/nonexistent/photo_{i}.jpg" for i in range(5)]

        result = processor.process_batch(bad_paths)

        assert result.total_photos == 5
        assert result.successful == 0
        assert result.failed == 5
        assert len(result.scores) == 0
        assert len(result.errors) == 5

    def test_process_batch_progress_callback(self, processor, test_photos):
        """Test progress callback during batch processing."""
        progress_updates = []

        def track_progress(analyzed, total, failed):
            progress_updates.append({
                'analyzed': analyzed,
                'total': total,
                'failed': failed
            })

        result = processor.process_batch(
            test_photos,
            progress_callback=track_progress
        )

        # Verify progress was tracked
        assert len(progress_updates) == 10  # One update per photo
        assert progress_updates[-1]['analyzed'] == 10
        assert progress_updates[-1]['total'] == 10
        assert progress_updates[-1]['failed'] == 0

    def test_process_batch_chunked_empty(self, processor):
        """Test chunked processing with empty list."""
        result = processor.process_batch_chunked([])

        assert result.total_photos == 0
        assert result.successful == 0
        assert result.failed == 0

    def test_process_batch_chunked_single_chunk(self, processor, test_photos):
        """Test chunked processing with single chunk."""
        # Chunk size larger than batch
        result = processor.process_batch_chunked(test_photos, chunk_size=100)

        assert result.total_photos == 10
        assert result.successful == 10
        assert result.failed == 0

    def test_process_batch_chunked_multiple_chunks(self, processor, test_photos):
        """Test chunked processing with multiple chunks."""
        # Small chunk size to force multiple chunks
        result = processor.process_batch_chunked(test_photos, chunk_size=3)

        assert result.total_photos == 10
        assert result.successful == 10
        assert result.failed == 0
        assert len(result.scores) == 10

    def test_process_batch_chunked_progress(self, processor, test_photos):
        """Test progress callback in chunked processing."""
        progress_updates = []

        def track_progress(analyzed, total, failed):
            progress_updates.append({
                'analyzed': analyzed,
                'total': total,
                'failed': failed
            })

        result = processor.process_batch_chunked(
            test_photos,
            chunk_size=3,
            progress_callback=track_progress
        )

        # Should have updates after each chunk (10 photos / chunk_size=3 = 4 chunks)
        assert len(progress_updates) == 4
        assert progress_updates[-1]['analyzed'] == 10
        assert progress_updates[-1]['total'] == 10

    def test_validate_paths_all_valid(self, processor, test_photos):
        """Test path validation with all valid paths."""
        valid, invalid = processor.validate_paths(test_photos)

        assert len(valid) == 10
        assert len(invalid) == 0

    def test_validate_paths_nonexistent(self, processor, test_photos):
        """Test path validation with non-existent files."""
        paths = test_photos + ["/nonexistent/photo.jpg"]

        valid, invalid = processor.validate_paths(paths)

        assert len(valid) == 10
        assert len(invalid) == 1
        assert invalid[0][0] == "/nonexistent/photo.jpg"
        assert "does not exist" in invalid[0][1]

    def test_validate_paths_empty_file(self, processor, temp_dir):
        """Test path validation with empty file."""
        empty_file = Path(temp_dir) / "empty.jpg"
        empty_file.touch()

        valid, invalid = processor.validate_paths([str(empty_file)])

        assert len(valid) == 0
        assert len(invalid) == 1
        assert "empty" in invalid[0][1].lower()

    def test_validate_paths_directory(self, processor, temp_dir):
        """Test path validation with directory instead of file."""
        valid, invalid = processor.validate_paths([temp_dir])

        assert len(valid) == 0
        assert len(invalid) == 1
        assert "Not a file" in invalid[0][1]

    def test_validate_paths_mixed(self, processor, test_photos, temp_dir):
        """Test path validation with mix of valid and invalid paths."""
        mixed_paths = [
            test_photos[0],
            "/nonexistent.jpg",
            test_photos[1],
            temp_dir,  # directory
        ]

        valid, invalid = processor.validate_paths(mixed_paths)

        assert len(valid) == 2
        assert len(invalid) == 2


class TestEmotionalBatchProcessorPerformance:
    """Test performance characteristics of batch processor."""

    def test_parallel_processing_completes(self, test_photos):
        """Test that parallel processing completes successfully."""
        import time

        # Create larger test set
        large_test_set = test_photos * 3  # 30 photos

        # Process with multiple workers
        processor = EmotionalBatchProcessor(num_workers=2)
        start = time.time()
        result = processor.process_batch(large_test_set)
        elapsed = time.time() - start

        # Verify processing completed
        assert result.successful == 30
        assert elapsed > 0
        print(f"Processed {result.successful} photos in {elapsed:.2f}s")

    def test_chunked_processing_memory_efficient(self, test_photos):
        """Test that chunked processing works for large batches."""
        # Simulate large batch
        large_batch = test_photos * 10  # 100 photos

        processor = EmotionalBatchProcessor(num_workers=2)
        result = processor.process_batch_chunked(large_batch, chunk_size=25)

        # Verify all photos processed
        assert result.total_photos == 100
        assert result.successful == 100
        assert result.failed == 0


class TestPathHandling:
    """Test Path object handling."""

    def test_process_with_path_objects(self, processor, test_photos):
        """Test processing with Path objects instead of strings."""
        path_objects = [Path(p) for p in test_photos]

        result = processor.process_batch(path_objects)

        assert result.successful == 10
        assert result.failed == 0

    def test_validate_with_path_objects(self, processor, test_photos):
        """Test validation with Path objects."""
        path_objects = [Path(p) for p in test_photos]

        valid, invalid = processor.validate_paths(path_objects)

        assert len(valid) == 10
        assert len(invalid) == 0


class TestScoreProperties:
    """Test that returned scores have expected properties."""

    def test_score_components(self, processor, test_photos):
        """Test that scores have all required components."""
        result = processor.process_batch([test_photos[0]])

        assert len(result.scores) == 1
        path, score = result.scores[0]

        # Verify score has all components
        assert hasattr(score, 'face_count')
        assert hasattr(score, 'emotion_score')
        assert hasattr(score, 'intimacy_score')
        assert hasattr(score, 'engagement_score')
        assert hasattr(score, 'composite')
        assert hasattr(score, 'tier')

        # Verify values are in expected ranges
        assert 0 <= score.composite <= 100
        assert score.tier in ['high', 'medium', 'low']

    def test_metadata_included(self, processor, test_photos):
        """Test that scores include metadata."""
        result = processor.process_batch([test_photos[0]])

        assert len(result.scores) == 1
        path, score = result.scores[0]

        assert score.metadata is not None
        assert isinstance(score.metadata, dict)
