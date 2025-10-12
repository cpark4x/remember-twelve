"""
Batch Processor for Emotional Significance Detector

Provides parallel batch processing of photos with progress tracking,
error handling, and configurable worker pools.

Key Features:
- Parallel processing using ProcessPoolExecutor
- Progress tracking with callbacks
- Graceful error handling (individual photo failures don't stop batch)
- Result aggregation
- Memory-efficient chunking

Examples:
    >>> from emotional_significance import EmotionalBatchProcessor
    >>> processor = EmotionalBatchProcessor()
    >>>
    >>> # Process batch with progress callback
    >>> def on_progress(analyzed, total, failed):
    ...     print(f"Progress: {analyzed}/{total} ({failed} failed)")
    >>>
    >>> results = processor.process_batch(
    ...     photo_paths=['photo1.jpg', 'photo2.jpg'],
    ...     progress_callback=on_progress
    ... )
"""

from typing import List, Optional, Callable, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path

from .analyzer import EmotionalAnalyzer
from .data_classes import EmotionalScore
from .config import EmotionalConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """
    Result of batch processing operation.

    Attributes:
        total_photos: Total number of photos attempted
        successful: Number of successfully analyzed photos
        failed: Number of failed photos
        scores: List of (photo_path, EmotionalScore) tuples for successful analyses
        errors: List of (photo_path, error_message) tuples for failed analyses
    """
    total_photos: int
    successful: int
    failed: int
    scores: List[Tuple[str, EmotionalScore]]
    errors: List[Tuple[str, str]]

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_photos == 0:
            return 0.0
        return (self.successful / self.total_photos) * 100


def _process_single_photo(photo_path: str, config_dict: dict) -> Tuple[str, Optional[EmotionalScore], Optional[str]]:
    """
    Process a single photo (worker function for multiprocessing).

    This function must be at module level for pickling by multiprocessing.

    Args:
        photo_path: Path to photo
        config_dict: Configuration dictionary (serializable)

    Returns:
        Tuple of (photo_path, EmotionalScore or None, error_message or None)
    """
    try:
        # Recreate analyzer in worker process
        from .config import EmotionalConfig
        from .analyzer import EmotionalAnalyzer

        # Recreate config from dict
        config = EmotionalConfig()
        # Apply config overrides from dict if needed
        # For now, use default config
        # TODO: If custom config needed, deserialize from config_dict

        analyzer = EmotionalAnalyzer(config=config)
        score = analyzer.analyze_photo(photo_path)

        return (photo_path, score, None)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.warning(f"Failed to analyze {photo_path}: {error_msg}")
        return (photo_path, None, error_msg)


class EmotionalBatchProcessor:
    """
    Batch processor for analyzing multiple photos in parallel.

    Uses ProcessPoolExecutor for CPU-bound analysis tasks. Provides
    progress tracking, error handling, and memory-efficient chunking.

    Examples:
        >>> processor = EmotionalBatchProcessor(num_workers=4)
        >>> result = processor.process_batch(photo_paths)
        >>> print(f"Analyzed {result.successful}/{result.total_photos} photos")
        >>> print(f"Success rate: {result.success_rate:.1f}%")
    """

    def __init__(
        self,
        num_workers: int = 4,
        config: Optional[EmotionalConfig] = None
    ):
        """
        Initialize batch processor.

        Args:
            num_workers: Number of parallel workers (defaults to 4)
            config: Emotional analyzer configuration (uses default if None)
        """
        from .config import get_default_config

        self.config = config or get_default_config()
        self.num_workers = num_workers
        self.chunk_size = self.config.performance.default_batch_size

        # Convert config to dict for serialization
        self.config_dict = self.config.to_dict()

    def process_batch(
        self,
        photo_paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int, int], None]] = None
    ) -> BatchResult:
        """
        Process a batch of photos in parallel.

        Args:
            photo_paths: List of photo file paths
            progress_callback: Optional callback(analyzed, total, failed) for progress updates

        Returns:
            BatchResult with scores and error information

        Examples:
            >>> def show_progress(analyzed, total, failed):
            ...     pct = (analyzed / total) * 100
            ...     print(f"Progress: {pct:.1f}% ({failed} failed)")
            >>>
            >>> result = processor.process_batch(
            ...     photo_paths=paths,
            ...     progress_callback=show_progress
            ... )
        """
        # Convert Path objects to strings
        photo_paths = [str(p) for p in photo_paths]

        total_photos = len(photo_paths)

        if total_photos == 0:
            return BatchResult(
                total_photos=0,
                successful=0,
                failed=0,
                scores=[],
                errors=[]
            )

        logger.info(f"Starting batch processing of {total_photos} photos with {self.num_workers} workers")

        scores: List[Tuple[str, EmotionalScore]] = []
        errors: List[Tuple[str, str]] = []
        analyzed_count = 0
        failed_count = 0

        # Process in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(_process_single_photo, path, self.config_dict): path
                for path in photo_paths
            }

            # Collect results as they complete
            for future in as_completed(futures):
                photo_path, score, error = future.result()
                analyzed_count += 1

                if score is not None:
                    scores.append((photo_path, score))
                else:
                    failed_count += 1
                    errors.append((photo_path, error or "Unknown error"))

                # Call progress callback if provided
                if progress_callback is not None:
                    progress_callback(analyzed_count, total_photos, failed_count)

        result = BatchResult(
            total_photos=total_photos,
            successful=len(scores),
            failed=failed_count,
            scores=scores,
            errors=errors
        )

        logger.info(
            f"Batch processing complete: {result.successful}/{result.total_photos} successful "
            f"({result.success_rate:.1f}%), {result.failed} failed"
        )

        return result

    def process_batch_chunked(
        self,
        photo_paths: List[Union[str, Path]],
        chunk_size: int = 1000,
        progress_callback: Optional[Callable[[int, int, int], None]] = None
    ) -> BatchResult:
        """
        Process a batch of photos in chunks to manage memory.

        For very large batches (>10,000 photos), processing in chunks
        helps keep memory usage under control.

        Args:
            photo_paths: List of photo file paths
            chunk_size: Photos per chunk (defaults to 1000)
            progress_callback: Optional callback(analyzed, total, failed) for progress updates

        Returns:
            BatchResult with scores and error information

        Examples:
            >>> # Process 100,000 photos in chunks of 1000
            >>> processor = EmotionalBatchProcessor()
            >>> result = processor.process_batch_chunked(
            ...     large_photo_list,
            ...     chunk_size=1000
            ... )
        """
        # Convert Path objects to strings
        photo_paths = [str(p) for p in photo_paths]

        total_photos = len(photo_paths)

        if total_photos == 0:
            return BatchResult(
                total_photos=0,
                successful=0,
                failed=0,
                scores=[],
                errors=[]
            )

        logger.info(f"Starting chunked batch processing of {total_photos} photos")
        logger.info(f"Using chunk size of {chunk_size} and {self.num_workers} workers")

        all_scores: List[Tuple[str, EmotionalScore]] = []
        all_errors: List[Tuple[str, str]] = []
        total_analyzed = 0
        total_failed = 0

        # Process in chunks
        for i in range(0, total_photos, chunk_size):
            chunk = photo_paths[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (total_photos + chunk_size - 1) // chunk_size

            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} photos)")

            # Process this chunk
            chunk_result = self.process_batch(
                photo_paths=chunk,
                progress_callback=None  # We'll aggregate progress at the end
            )

            all_scores.extend(chunk_result.scores)
            all_errors.extend(chunk_result.errors)
            total_analyzed += len(chunk)
            total_failed += chunk_result.failed

            # Call progress callback after each chunk
            if progress_callback is not None:
                progress_callback(total_analyzed, total_photos, total_failed)

        result = BatchResult(
            total_photos=total_photos,
            successful=len(all_scores),
            failed=total_failed,
            scores=all_scores,
            errors=all_errors
        )

        logger.info(
            f"Chunked batch processing complete: {result.successful}/{result.total_photos} successful "
            f"({result.success_rate:.1f}%), {result.failed} failed"
        )

        return result

    def validate_paths(self, photo_paths: List[Union[str, Path]]) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Validate that photo paths exist and are readable.

        Args:
            photo_paths: List of photo file paths

        Returns:
            Tuple of (valid_paths, invalid_path_errors)

        Examples:
            >>> valid, invalid = processor.validate_paths(photo_paths)
            >>> if invalid:
            ...     print(f"Warning: {len(invalid)} invalid paths")
            >>> result = processor.process_batch(valid)
        """
        valid_paths: List[str] = []
        invalid_paths: List[Tuple[str, str]] = []

        for path in photo_paths:
            path_str = str(path)
            path_obj = Path(path_str)

            if not path_obj.exists():
                invalid_paths.append((path_str, "File does not exist"))
            elif not path_obj.is_file():
                invalid_paths.append((path_str, "Not a file"))
            elif not path_obj.stat().st_size > 0:
                invalid_paths.append((path_str, "File is empty"))
            else:
                valid_paths.append(path_str)

        if invalid_paths:
            logger.warning(f"Found {len(invalid_paths)} invalid paths out of {len(photo_paths)} total")

        return valid_paths, invalid_paths
