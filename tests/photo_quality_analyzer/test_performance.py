"""
Tests for PerformanceMonitor

Covers performance tracking, memory monitoring, statistics calculation,
context manager usage, and aggregate monitoring.
"""

import pytest
import time

from src.photo_quality_analyzer.performance import (
    PerformanceMonitor,
    PerformanceMetrics,
    AggregateMonitor,
    track_performance
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            start_time=time.time(),
            duration=10.5,
            photos_processed=100,
            photos_per_second=9.5
        )

        assert metrics.duration == 10.5
        assert metrics.photos_processed == 100
        assert metrics.photos_per_second == 9.5

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(
            start_time=time.time(),
            duration=10.0,
            photos_processed=50,
            photos_per_second=5.0,
            memory_start_mb=100.0,
            memory_peak_mb=150.0,
            memory_end_mb=120.0
        )

        result = metrics.to_dict()

        assert result['duration'] == 10.0
        assert result['photos_processed'] == 50
        assert result['photos_per_second'] == 5.0
        assert result['memory_delta_mb'] == 20.0  # 120 - 100


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(name="test")

        assert monitor.name == "test"
        assert not monitor._is_running
        assert monitor._metrics is None

    def test_start_stop(self):
        """Test starting and stopping monitor."""
        monitor = PerformanceMonitor()

        monitor.start()
        assert monitor._is_running

        time.sleep(0.1)  # Short delay

        metrics = monitor.stop()
        assert not monitor._is_running
        assert metrics.duration > 0
        assert metrics.duration >= 0.1

    def test_start_already_running(self):
        """Test starting monitor when already running."""
        monitor = PerformanceMonitor()

        monitor.start()
        # Start again (should log warning but not crash)
        monitor.start()

        assert monitor._is_running

        monitor.stop()

    def test_stop_not_running(self):
        """Test stopping monitor when not running."""
        monitor = PerformanceMonitor()

        # Stop without starting (should log warning but not crash)
        metrics = monitor.stop()

        assert not monitor._is_running

    def test_record_photo(self):
        """Test recording photo processing."""
        monitor = PerformanceMonitor()

        monitor.start()
        monitor.record_photo()
        monitor.record_photo()
        monitor.record_photo()

        metrics = monitor.stop()

        assert metrics.photos_processed == 3

    def test_record_multiple_photos(self):
        """Test recording multiple photos at once."""
        monitor = PerformanceMonitor()

        monitor.start()
        monitor.record_photo(count=10)
        monitor.record_photo(count=5)

        metrics = monitor.stop()

        assert metrics.photos_processed == 15

    def test_photos_per_second_calculation(self):
        """Test photos per second calculation."""
        monitor = PerformanceMonitor()

        monitor.start()

        # Simulate processing 10 photos over ~0.1 seconds
        for _ in range(10):
            monitor.record_photo()
            time.sleep(0.01)

        metrics = monitor.stop()

        # Should be roughly 100 photos/sec (10 photos in 0.1 sec)
        # Allow for some variance
        assert metrics.photos_per_second > 50  # At least 50/sec
        assert metrics.photos_per_second < 200  # Less than 200/sec

    def test_get_current_rate(self):
        """Test getting current processing rate."""
        monitor = PerformanceMonitor()

        monitor.start()
        monitor.record_photo(count=10)
        time.sleep(0.1)

        rate = monitor.get_current_rate()

        # Should have some positive rate
        assert rate > 0

        monitor.stop()

    def test_get_current_rate_not_running(self):
        """Test getting rate when not running."""
        monitor = PerformanceMonitor()

        rate = monitor.get_current_rate()

        assert rate == 0.0

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        monitor = PerformanceMonitor()

        monitor.start()

        # Allocate some memory
        big_list = [0] * 1000000  # Allocate ~8MB

        monitor.record_photo()

        metrics = monitor.stop()

        # Should have tracked some memory
        assert metrics.memory_start_mb > 0
        assert metrics.memory_end_mb > 0
        assert metrics.memory_peak_mb >= metrics.memory_start_mb

        # Clean up
        del big_list

    def test_get_stats_not_started(self):
        """Test getting stats before starting."""
        monitor = PerformanceMonitor()

        stats = monitor.get_stats()

        assert stats['duration'] == 0.0
        assert stats['photos_processed'] == 0
        assert stats['photos_per_second'] == 0.0

    def test_get_stats_while_running(self):
        """Test getting stats while running."""
        monitor = PerformanceMonitor()

        monitor.start()
        monitor.record_photo(count=5)
        time.sleep(0.1)

        stats = monitor.get_stats()

        assert stats['duration'] > 0
        assert stats['photos_processed'] == 5
        assert stats['photos_per_second'] > 0

        monitor.stop()

    def test_get_stats_after_stop(self):
        """Test getting stats after stopping."""
        monitor = PerformanceMonitor()

        monitor.start()
        monitor.record_photo(count=10)
        time.sleep(0.1)
        monitor.stop()

        stats = monitor.get_stats()

        assert stats['duration'] > 0
        assert stats['photos_processed'] == 10
        assert stats['photos_per_second'] > 0
        assert 'memory_delta_mb' in stats


class TestPerformanceMonitorContextManager:
    """Test PerformanceMonitor as context manager."""

    def test_context_manager_usage(self):
        """Test using monitor as context manager."""
        monitor = PerformanceMonitor()

        with monitor:
            assert monitor._is_running
            monitor.record_photo(count=5)

        # Should auto-stop
        assert not monitor._is_running

        stats = monitor.get_stats()
        assert stats['photos_processed'] == 5

    def test_context_manager_with_exception(self):
        """Test context manager behavior with exception."""
        monitor = PerformanceMonitor()

        try:
            with monitor:
                monitor.record_photo(count=3)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should still have stopped
        assert not monitor._is_running

        stats = monitor.get_stats()
        assert stats['photos_processed'] == 3

    def test_nested_operations(self):
        """Test nested monitoring operations."""
        with PerformanceMonitor() as outer:
            outer.record_photo(count=10)

            # Simulate some work
            time.sleep(0.05)

            outer.record_photo(count=10)

        outer_stats = outer.get_stats()
        assert outer_stats['photos_processed'] == 20


class TestAggregateMonitor:
    """Test AggregateMonitor class."""

    def test_aggregate_initialization(self):
        """Test aggregate monitor initialization."""
        aggregate = AggregateMonitor()

        summary = aggregate.get_summary()

        assert summary['num_sessions'] == 0
        assert summary['total_duration'] == 0.0
        assert summary['total_photos'] == 0

    def test_add_single_session(self):
        """Test adding a single session."""
        aggregate = AggregateMonitor()

        with PerformanceMonitor() as monitor:
            monitor.record_photo(count=10)
            time.sleep(0.1)

        aggregate.add_session(monitor.get_stats())

        summary = aggregate.get_summary()

        assert summary['num_sessions'] == 1
        assert summary['total_photos'] == 10
        assert summary['total_duration'] > 0

    def test_add_multiple_sessions(self):
        """Test adding multiple sessions."""
        aggregate = AggregateMonitor()

        # Add 3 sessions
        for i in range(3):
            with PerformanceMonitor() as monitor:
                monitor.record_photo(count=10 * (i + 1))
                time.sleep(0.05)

            aggregate.add_session(monitor.get_stats())

        summary = aggregate.get_summary()

        assert summary['num_sessions'] == 3
        assert summary['total_photos'] == 10 + 20 + 30  # 60 total
        assert summary['avg_photos_per_second'] > 0

    def test_aggregate_memory_tracking(self):
        """Test aggregate memory tracking."""
        aggregate = AggregateMonitor()

        for _ in range(2):
            with PerformanceMonitor() as monitor:
                monitor.record_photo(count=5)

            aggregate.add_session(monitor.get_stats())

        summary = aggregate.get_summary()

        assert 'peak_memory_mb' in summary
        assert 'total_memory_delta_mb' in summary

    def test_aggregate_average_rate(self):
        """Test average rate calculation."""
        aggregate = AggregateMonitor()

        # Create sessions with known rates
        for _ in range(3):
            with PerformanceMonitor() as monitor:
                monitor.record_photo(count=10)
                time.sleep(0.1)

            aggregate.add_session(monitor.get_stats())

        summary = aggregate.get_summary()

        # Total 30 photos, roughly 0.3 seconds = ~100 photos/sec
        assert summary['avg_photos_per_second'] > 50


class TestTrackPerformanceFunction:
    """Test track_performance convenience function."""

    def test_track_performance_basic(self, capsys):
        """Test basic usage of track_performance."""
        with track_performance("test_operation") as monitor:
            monitor.record_photo(count=5)
            time.sleep(0.05)

        # Should have printed summary
        captured = capsys.readouterr()
        assert "Performance Summary" in captured.out
        assert "test_operation" in captured.out

    def test_track_performance_with_exception(self, capsys):
        """Test track_performance with exception."""
        try:
            with track_performance("test_error") as monitor:
                monitor.record_photo(count=3)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should still print summary
        captured = capsys.readouterr()
        assert "Performance Summary" in captured.out


class TestPerformanceSummaryPrinting:
    """Test summary printing methods."""

    def test_print_summary(self, capsys):
        """Test print_summary method."""
        monitor = PerformanceMonitor(name="test_print")

        with monitor:
            monitor.record_photo(count=10)
            time.sleep(0.1)

        monitor.print_summary()

        captured = capsys.readouterr()
        output = captured.out

        assert "Performance Summary" in output
        assert "test_print" in output
        assert "Duration:" in output
        assert "Photos processed: 10" in output
        assert "Rate:" in output
        assert "Memory:" in output

    def test_aggregate_print_summary(self, capsys):
        """Test aggregate print_summary method."""
        aggregate = AggregateMonitor()

        for i in range(3):
            with PerformanceMonitor() as monitor:
                monitor.record_photo(count=10)
                time.sleep(0.05)

            aggregate.add_session(monitor.get_stats())

        aggregate.print_summary()

        captured = capsys.readouterr()
        output = captured.out

        assert "Aggregate Performance Summary" in output
        assert "Sessions: 3" in output
        assert "Total photos: 30" in output
        assert "Average rate:" in output


class TestPerformanceEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_duration(self):
        """Test handling of zero duration (very fast operation)."""
        monitor = PerformanceMonitor()

        monitor.start()
        monitor.record_photo(count=5)
        metrics = monitor.stop()  # Immediate stop

        # Should handle gracefully (rate might be very high or infinity)
        assert metrics.photos_processed == 5
        # Rate calculation should not crash
        assert metrics.photos_per_second >= 0

    def test_no_photos_processed(self):
        """Test monitoring with no photos processed."""
        monitor = PerformanceMonitor()

        monitor.start()
        time.sleep(0.1)
        metrics = monitor.stop()

        assert metrics.photos_processed == 0
        assert metrics.photos_per_second == 0.0

    def test_memory_error_handling(self):
        """Test handling of memory monitoring errors."""
        monitor = PerformanceMonitor()

        # Should handle gracefully even if memory monitoring fails
        monitor.start()
        monitor.record_photo()

        # Get memory should not crash
        memory = monitor._get_memory_usage_mb()
        assert memory >= 0  # Should return 0 or actual value

        monitor.stop()

    def test_large_number_of_photos(self):
        """Test handling large number of photos."""
        monitor = PerformanceMonitor()

        monitor.start()
        monitor.record_photo(count=1000000)  # 1 million photos
        time.sleep(0.1)
        metrics = monitor.stop()

        assert metrics.photos_processed == 1000000
        assert metrics.photos_per_second > 0

    def test_very_long_duration(self):
        """Test monitoring over longer duration."""
        monitor = PerformanceMonitor()

        monitor.start()
        monitor.record_photo(count=10)
        time.sleep(0.3)  # Longer duration
        metrics = monitor.stop()

        assert metrics.duration >= 0.3
        assert metrics.photos_processed == 10


class TestPerformanceIntegration:
    """Test performance monitoring in realistic scenarios."""

    def test_batch_processing_simulation(self):
        """Test monitoring a batch processing scenario."""
        aggregate = AggregateMonitor()

        # Simulate processing 5 batches
        for batch_num in range(5):
            with PerformanceMonitor(name=f"batch_{batch_num}") as monitor:
                # Simulate processing 100 photos per batch
                for _ in range(100):
                    monitor.record_photo()
                    time.sleep(0.001)  # 1ms per photo

            aggregate.add_session(monitor.get_stats())

        summary = aggregate.get_summary()

        assert summary['num_sessions'] == 5
        assert summary['total_photos'] == 500
        assert summary['avg_photos_per_second'] > 0

    def test_progressive_rate_checking(self):
        """Test checking rate progressively during processing."""
        monitor = PerformanceMonitor()
        rates = []

        with monitor:
            for i in range(50):
                monitor.record_photo()
                time.sleep(0.01)

                if i % 10 == 0:
                    rates.append(monitor.get_current_rate())

        # Should have captured multiple rate samples
        assert len(rates) > 0
        # Rates should be positive
        assert all(rate > 0 for rate in rates)

    def test_memory_intensive_operation(self):
        """Test monitoring memory-intensive operation."""
        monitor = PerformanceMonitor()

        with monitor:
            # Allocate memory in chunks
            data = []
            for _ in range(10):
                data.append([0] * 100000)  # ~800KB per iteration
                monitor.record_photo()
                time.sleep(0.01)

        stats = monitor.get_stats()

        # Should show memory increase
        assert stats['memory_delta_mb'] != 0  # Some memory change

        # Clean up
        del data
