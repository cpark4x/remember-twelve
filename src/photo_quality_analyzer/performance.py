"""
Performance Monitor for Photo Quality Analyzer

Provides performance tracking and monitoring for analysis operations.
Uses context manager pattern for clean tracking.

Key Features:
- Processing time tracking
- Memory usage monitoring
- Photos per second calculation
- Aggregate statistics across batches
- Context manager for automatic start/stop

Examples:
    >>> from photo_quality_analyzer import PerformanceMonitor
    >>>
    >>> with PerformanceMonitor() as monitor:
    ...     # Analyze photos
    ...     for photo in photos:
    ...         analyzer.analyze_photo(photo)
    ...         monitor.record_photo()
    >>>
    >>> print(f"Processed {monitor.photos_per_second:.1f} photos/sec")
"""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import psutil
import os


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a monitoring session.

    Attributes:
        start_time: Start timestamp
        end_time: End timestamp (None if still running)
        duration: Total duration in seconds
        photos_processed: Number of photos processed
        photos_per_second: Processing rate
        memory_start_mb: Memory usage at start (MB)
        memory_peak_mb: Peak memory usage (MB)
        memory_end_mb: Memory usage at end (MB)
    """
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    photos_processed: int = 0
    photos_per_second: float = 0.0
    memory_start_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_end_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'duration': self.duration,
            'photos_processed': self.photos_processed,
            'photos_per_second': self.photos_per_second,
            'memory_start_mb': self.memory_start_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_end_mb': self.memory_end_mb,
            'memory_delta_mb': self.memory_end_mb - self.memory_start_mb
        }


class PerformanceMonitor:
    """
    Monitor performance of photo analysis operations.

    Use as a context manager to automatically track timing and memory.
    Provides methods to record progress and calculate statistics.

    Examples:
        >>> monitor = PerformanceMonitor()
        >>> with monitor:
        ...     for photo in photos:
        ...         score = analyzer.analyze_photo(photo)
        ...         monitor.record_photo()
        >>>
        >>> stats = monitor.get_stats()
        >>> print(f"Duration: {stats['duration']:.2f}s")
        >>> print(f"Rate: {stats['photos_per_second']:.1f} photos/sec")
    """

    def __init__(self, name: str = "analysis"):
        """
        Initialize performance monitor.

        Args:
            name: Name for this monitoring session (for logging)
        """
        self.name = name
        self._metrics: Optional[PerformanceMetrics] = None
        self._is_running = False
        self._process = psutil.Process(os.getpid())

    def start(self) -> None:
        """
        Start monitoring.

        Examples:
            >>> monitor = PerformanceMonitor()
            >>> monitor.start()
            >>> # ... do work ...
            >>> monitor.stop()
        """
        if self._is_running:
            logger.warning("Monitor already running")
            return

        memory_mb = self._get_memory_usage_mb()

        self._metrics = PerformanceMetrics(
            start_time=time.time(),
            memory_start_mb=memory_mb,
            memory_peak_mb=memory_mb
        )

        self._is_running = True
        logger.info(f"Performance monitor '{self.name}' started")

    def stop(self) -> PerformanceMetrics:
        """
        Stop monitoring and return metrics.

        Returns:
            PerformanceMetrics with final statistics

        Examples:
            >>> monitor.start()
            >>> # ... do work ...
            >>> metrics = monitor.stop()
            >>> print(f"Processed {metrics.photos_processed} photos")
        """
        if not self._is_running:
            logger.warning("Monitor not running")
            return PerformanceMetrics(start_time=time.time())

        self._metrics.end_time = time.time()
        self._metrics.duration = self._metrics.end_time - self._metrics.start_time
        self._metrics.memory_end_mb = self._get_memory_usage_mb()

        # Calculate photos per second
        if self._metrics.duration > 0:
            self._metrics.photos_per_second = (
                self._metrics.photos_processed / self._metrics.duration
            )

        self._is_running = False

        logger.info(
            f"Performance monitor '{self.name}' stopped: "
            f"{self._metrics.photos_processed} photos in {self._metrics.duration:.2f}s "
            f"({self._metrics.photos_per_second:.1f} photos/sec)"
        )

        return self._metrics

    def record_photo(self, count: int = 1) -> None:
        """
        Record that photo(s) have been processed.

        Args:
            count: Number of photos to record (default 1)

        Examples:
            >>> with monitor:
            ...     for photo in photos:
            ...         analyzer.analyze_photo(photo)
            ...         monitor.record_photo()
        """
        if not self._is_running:
            return

        self._metrics.photos_processed += count

        # Update peak memory
        current_memory = self._get_memory_usage_mb()
        if current_memory > self._metrics.memory_peak_mb:
            self._metrics.memory_peak_mb = current_memory

    def get_current_rate(self) -> float:
        """
        Get current photos per second rate.

        Returns:
            Current processing rate (photos/sec)

        Examples:
            >>> with monitor:
            ...     for i, photo in enumerate(photos):
            ...         analyzer.analyze_photo(photo)
            ...         monitor.record_photo()
            ...         if i % 100 == 0:
            ...             print(f"Current rate: {monitor.get_current_rate():.1f} photos/sec")
        """
        if not self._is_running or not self._metrics:
            return 0.0

        elapsed = time.time() - self._metrics.start_time
        if elapsed > 0:
            return self._metrics.photos_processed / elapsed
        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with performance statistics

        Examples:
            >>> with monitor:
            ...     # ... process photos ...
            ...     pass
            >>> stats = monitor.get_stats()
            >>> print(f"Memory used: {stats['memory_delta_mb']:.1f} MB")
        """
        if not self._metrics:
            return {
                'duration': 0.0,
                'photos_processed': 0,
                'photos_per_second': 0.0,
                'memory_start_mb': 0.0,
                'memory_peak_mb': 0.0,
                'memory_end_mb': 0.0,
                'memory_delta_mb': 0.0
            }

        # If still running, calculate current duration
        if self._is_running:
            current_duration = time.time() - self._metrics.start_time
            current_rate = (
                self._metrics.photos_processed / current_duration
                if current_duration > 0 else 0.0
            )

            return {
                'duration': current_duration,
                'photos_processed': self._metrics.photos_processed,
                'photos_per_second': current_rate,
                'memory_start_mb': self._metrics.memory_start_mb,
                'memory_peak_mb': self._metrics.memory_peak_mb,
                'memory_end_mb': self._get_memory_usage_mb(),
                'memory_delta_mb': self._get_memory_usage_mb() - self._metrics.memory_start_mb
            }

        return self._metrics.to_dict()

    def _get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        try:
            memory_bytes = self._process.memory_info().rss
            return memory_bytes / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def print_summary(self) -> None:
        """
        Print performance summary to console.

        Examples:
            >>> with monitor:
            ...     # ... process photos ...
            ...     pass
            >>> monitor.print_summary()
            Performance Summary (analysis):
            - Duration: 45.23s
            - Photos processed: 1000
            - Rate: 22.1 photos/sec
            - Memory: 125.3 MB -> 245.7 MB (delta: +120.4 MB)
            - Peak memory: 267.8 MB
        """
        stats = self.get_stats()

        print(f"\nPerformance Summary ({self.name}):")
        print(f"- Duration: {stats['duration']:.2f}s")
        print(f"- Photos processed: {stats['photos_processed']}")
        print(f"- Rate: {stats['photos_per_second']:.1f} photos/sec")
        print(
            f"- Memory: {stats['memory_start_mb']:.1f} MB -> "
            f"{stats['memory_end_mb']:.1f} MB "
            f"(delta: {stats['memory_delta_mb']:+.1f} MB)"
        )
        print(f"- Peak memory: {stats['memory_peak_mb']:.1f} MB")


class AggregateMonitor:
    """
    Aggregate performance statistics across multiple monitoring sessions.

    Useful for tracking performance over multiple batches or operations.

    Examples:
        >>> aggregate = AggregateMonitor()
        >>>
        >>> for batch in batches:
        ...     with PerformanceMonitor() as monitor:
        ...         # Process batch
        ...         pass
        ...     aggregate.add_session(monitor.get_stats())
        >>>
        >>> summary = aggregate.get_summary()
        >>> print(f"Total photos: {summary['total_photos']}")
        >>> print(f"Average rate: {summary['avg_photos_per_second']:.1f} photos/sec")
    """

    def __init__(self):
        """Initialize aggregate monitor."""
        self._sessions: List[Dict[str, Any]] = []

    def add_session(self, stats: Dict[str, Any]) -> None:
        """
        Add performance statistics from a session.

        Args:
            stats: Statistics dictionary from PerformanceMonitor.get_stats()

        Examples:
            >>> aggregate = AggregateMonitor()
            >>> with PerformanceMonitor() as monitor:
            ...     # ... process photos ...
            ...     pass
            >>> aggregate.add_session(monitor.get_stats())
        """
        self._sessions.append(stats)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get aggregate summary statistics.

        Returns:
            Dictionary with aggregate statistics

        Examples:
            >>> summary = aggregate.get_summary()
            >>> print(f"Total time: {summary['total_duration']:.2f}s")
            >>> print(f"Total photos: {summary['total_photos']}")
            >>> print(f"Average rate: {summary['avg_photos_per_second']:.1f} photos/sec")
        """
        if not self._sessions:
            return {
                'num_sessions': 0,
                'total_duration': 0.0,
                'total_photos': 0,
                'avg_photos_per_second': 0.0,
                'peak_memory_mb': 0.0,
                'total_memory_delta_mb': 0.0
            }

        total_duration = sum(s['duration'] for s in self._sessions)
        total_photos = sum(s['photos_processed'] for s in self._sessions)
        peak_memory = max(s['memory_peak_mb'] for s in self._sessions)
        total_memory_delta = sum(s['memory_delta_mb'] for s in self._sessions)

        avg_rate = total_photos / total_duration if total_duration > 0 else 0.0

        return {
            'num_sessions': len(self._sessions),
            'total_duration': total_duration,
            'total_photos': total_photos,
            'avg_photos_per_second': avg_rate,
            'peak_memory_mb': peak_memory,
            'total_memory_delta_mb': total_memory_delta
        }

    def print_summary(self) -> None:
        """
        Print aggregate summary to console.

        Examples:
            >>> aggregate.print_summary()
            Aggregate Performance Summary:
            - Sessions: 10
            - Total duration: 452.3s
            - Total photos: 10000
            - Average rate: 22.1 photos/sec
            - Peak memory: 267.8 MB
            - Total memory delta: +450.2 MB
        """
        summary = self.get_summary()

        print("\nAggregate Performance Summary:")
        print(f"- Sessions: {summary['num_sessions']}")
        print(f"- Total duration: {summary['total_duration']:.2f}s")
        print(f"- Total photos: {summary['total_photos']}")
        print(f"- Average rate: {summary['avg_photos_per_second']:.1f} photos/sec")
        print(f"- Peak memory: {summary['peak_memory_mb']:.1f} MB")
        print(f"- Total memory delta: {summary['total_memory_delta_mb']:+.1f} MB")


@contextmanager
def track_performance(name: str = "operation"):
    """
    Context manager for quick performance tracking.

    Convenience function for one-off monitoring.

    Args:
        name: Name for the operation being tracked

    Yields:
        PerformanceMonitor instance

    Examples:
        >>> with track_performance("batch_analysis") as monitor:
        ...     for photo in photos:
        ...         analyzer.analyze_photo(photo)
        ...         monitor.record_photo()
        >>> # Automatically prints summary on exit
    """
    monitor = PerformanceMonitor(name=name)
    monitor.start()
    try:
        yield monitor
    finally:
        monitor.stop()
        monitor.print_summary()
