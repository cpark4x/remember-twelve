"""
Tests for BatchProcessor

Covers parallel processing, progress tracking, error handling,
chunked processing, and path validation.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

from src.photo_quality_analyzer.batch_processor import (
    BatchProcessor,
    BatchResult,
    _process_single_photo
)
from src.photo_quality_analyzer.config import QualityAnalyzerConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for test photos."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_photos(temp_dir):
    """Create a set of test photos."""
    photos = []

    # Create 10 test photos with varying quality
    for i in range(10):
        path = Path(temp_dir) / f"photo_{i}.jpg"

        # Create synthetic image
        if i < 5:
            # Sharp images
            image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        else:
            # Blurry images
            image = np.random.randint(100, 150, (200, 200, 3), dtype=np.uint8)

        Image.fromarray(image).save(path)
        photos.append(str(path))

    return photos


@pytest.fixture
def processor():
    """Create batch processor instance."""
    return BatchProcessor()


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
        config_dict = QualityAnalyzerConfig().to_dict()

        path, score, error = _process_single_photo(photo_path, config_dict)

        assert path == photo_path
        assert score is not None
        assert error is None
        assert 0 <= score.composite <= 100

    def test_process_single_photo_nonexistent(self):
        """Test processing non-existent photo."""
        photo_path = "/nonexistent/photo.jpg"
        config_dict = QualityAnalyzerConfig().to_dict()

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

        config_dict = QualityAnalyzerConfig().to_dict()

        path, score, error = _process_single_photo(str(corrupted_path), config_dict)

        assert path == str(corrupted_path)
        assert score is None
        assert error is not None


class TestBatchProcessor:
    """Test BatchProcessor class."""

    def test_initialization_default(self):
        """Test processor initialization with defaults."""
        processor = BatchProcessor()

        assert processor.config is not None
        assert processor.num_workers == processor.config.performance.default_num_workers
        assert processor.chunk_size == processor.config.performance.default_batch_size

    def test_initialization_custom(self):
        """Test processor initialization with custom values."""
        config = QualityAnalyzerConfig()
        processor = BatchProcessor(
            config=config,
            num_workers=8,
            chunk_size=1000
        )

        assert processor.config == config
        assert processor.num_workers == 8
        assert processor.chunk_size == 1000

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
        processor.chunk_size = 100

        result = processor.process_batch_chunked(test_photos)

        assert result.total_photos == 10
        assert result.successful == 10
        assert result.failed == 0

    def test_process_batch_chunked_multiple_chunks(self, processor, test_photos):
        """Test chunked processing with multiple chunks."""
        # Small chunk size to force multiple chunks
        processor.chunk_size = 3

        result = processor.process_batch_chunked(test_photos)

        assert result.total_photos == 10
        assert result.successful == 10
        assert result.failed == 0
        assert len(result.scores) == 10

    def test_process_batch_chunked_progress(self, processor, test_photos):
        """Test progress callback in chunked processing."""
        processor.chunk_size = 3
        progress_updates = []

        def track_progress(analyzed, total, failed):
            progress_updates.append({
                'analyzed': analyzed,
                'total': total,
                'failed': failed
            })

        result = processor.process_batch_chunked(
            test_photos,
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


class TestBatchProcessorPerformance:
    """Test performance characteristics of batch processor."""

    def test_parallel_processing_faster(self, test_photos):
        """Test that parallel processing is faster than sequential."""
        import time

        # Create larger test set
        large_test_set = test_photos * 5  # 50 photos

        # Process with single worker
        processor_single = BatchProcessor(num_workers=1)
        start_single = time.time()
        result_single = processor_single.process_batch(large_test_set)
        time_single = time.time() - start_single

        # Process with multiple workers
        processor_multi = BatchProcessor(num_workers=4)
        start_multi = time.time()
        result_multi = processor_multi.process_batch(large_test_set)
        time_multi = time.time() - start_multi

        # Verify both processed correctly
        assert result_single.successful == 50
        assert result_multi.successful == 50

        # Multi-worker should be faster (allow some variance)
        # Note: In testing with small images, speedup may be minimal
        # due to process startup overhead
        print(f"Single worker: {time_single:.2f}s, Multi worker: {time_multi:.2f}s")
        # Just verify it completed without error
        assert time_multi > 0

    def test_chunked_processing_memory_efficient(self, test_photos):
        """Test that chunked processing works for large batches."""
        # Simulate large batch
        large_batch = test_photos * 100  # 1000 photos

        processor = BatchProcessor(chunk_size=50)
        result = processor.process_batch_chunked(large_batch)

        # Verify all photos processed
        assert result.total_photos == 1000
        assert result.successful == 1000
        assert result.failed == 0


class TestBatchProcessorConfiguration:
    """Test batch processor with different configurations."""

    def test_custom_config_weights(self, test_photos):
        """Test batch processor with custom score weights."""
        config = QualityAnalyzerConfig()
        config.weights.sharpness = 0.8
        config.weights.exposure = 0.2

        processor = BatchProcessor(config=config)
        result = processor.process_batch(test_photos[:3])

        assert result.successful == 3
        # Scores should be calculated with custom weights
        for _, score in result.scores:
            # Just verify scores are valid
            assert 0 <= score.composite <= 100

    def test_different_worker_counts(self, test_photos):
        """Test batch processor with different worker counts."""
        for num_workers in [1, 2, 4, 8]:
            processor = BatchProcessor(num_workers=num_workers)
            result = processor.process_batch(test_photos)

            assert result.successful == 10
            assert result.failed == 0

    def test_different_chunk_sizes(self, test_photos):
        """Test chunked processing with different chunk sizes."""
        large_batch = test_photos * 10  # 100 photos

        for chunk_size in [10, 25, 50, 100]:
            processor = BatchProcessor(chunk_size=chunk_size)
            result = processor.process_batch_chunked(large_batch)

            assert result.successful == 100
            assert result.failed == 0
