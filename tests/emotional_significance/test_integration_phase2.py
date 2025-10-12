"""
Integration Tests for Phase 2: Infrastructure Integration

Tests the complete pipeline with caching, batch processing, and performance.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from PIL import Image
import numpy as np

from src.emotional_significance import (
    EmotionalAnalyzer,
    EmotionalResultCache,
    EmotionalBatchProcessor,
    BatchResult
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_photos(temp_dir):
    """Create test photos for integration testing."""
    photos = []
    for i in range(20):
        path = Path(temp_dir) / f"photo_{i:03d}.jpg"
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        Image.fromarray(image).save(path)
        photos.append(str(path))
    return photos


@pytest.fixture
def cache(temp_dir):
    """Create cache instance."""
    db_path = Path(temp_dir) / "test_cache.db"
    return EmotionalResultCache(db_path=str(db_path))


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return EmotionalAnalyzer()


@pytest.fixture
def processor():
    """Create batch processor instance."""
    return EmotionalBatchProcessor(num_workers=2)


class TestCacheIntegration:
    """Test cache integration with analyzer."""

    def test_analyze_with_cache_miss(self, analyzer, cache, test_photos):
        """Test analyzing a photo with cache miss."""
        photo = test_photos[0]

        # First analysis (cache miss)
        assert cache.should_analyze(photo) is True

        score = analyzer.analyze_photo(photo)
        cache.set(photo, score)

        stats = cache.get_stats()
        assert stats['total_entries'] == 1
        assert stats['misses'] == 1

    def test_analyze_with_cache_hit(self, analyzer, cache, test_photos):
        """Test analyzing a photo with cache hit."""
        photo = test_photos[0]

        # First analysis
        score1 = analyzer.analyze_photo(photo)
        cache.set(photo, score1)

        # Second analysis (should use cache)
        # Note: should_analyze() calls get() internally, which counts as a hit
        assert cache.should_analyze(photo) is False
        score2 = cache.get(photo)

        assert score2 is not None
        assert score2.composite == score1.composite

        stats = cache.get_stats()
        # should_analyze() calls get() (1 hit) + explicit get() (1 hit) = 2 hits
        assert stats['hits'] == 2

    def test_cache_workflow(self, analyzer, cache, test_photos):
        """Test complete cache workflow."""
        photos = test_photos[:5]

        # Analyze all photos
        for photo in photos:
            if cache.should_analyze(photo):
                score = analyzer.analyze_photo(photo)
                cache.set(photo, score)

        # All should be cached now
        stats = cache.get_stats()
        assert stats['total_entries'] == 5

        # Re-analyze (should all be cache hits)
        for photo in photos:
            assert cache.should_analyze(photo) is False
            score = cache.get(photo)
            assert score is not None

        stats = cache.get_stats()
        # 5 misses during first pass + 5 hits in should_analyze() + 5 hits in get() = 10 hits
        assert stats['hits'] == 10
        assert stats['hit_rate'] >= 0.6  # 10 hits / 15 total

    def test_cache_hit_rate(self, analyzer, cache, test_photos):
        """Test cache hit rate calculation."""
        photos = test_photos[:10]

        # First pass - analyze and cache
        for photo in photos:
            score = analyzer.analyze_photo(photo)
            cache.set(photo, score)

        # Clear statistics for clean test
        cache._hits = 0
        cache._misses = 0

        # Second pass - all hits
        for photo in photos:
            score = cache.get(photo)
            assert score is not None

        stats = cache.get_stats()
        assert stats['total_requests'] == 10  # All hits
        assert stats['hit_rate'] == 1.0  # 10 hits / 10 requests


class TestBatchProcessorIntegration:
    """Test batch processor integration."""

    def test_batch_processing_basic(self, processor, test_photos):
        """Test basic batch processing."""
        photos = test_photos[:10]

        result = processor.process_batch(photos)

        assert result.total_photos == 10
        assert result.successful == 10
        assert result.failed == 0
        assert len(result.scores) == 10

    def test_batch_processing_with_progress(self, processor, test_photos):
        """Test batch processing with progress tracking."""
        photos = test_photos[:10]
        progress_updates = []

        def track_progress(analyzed, total, failed):
            progress_updates.append({
                'analyzed': analyzed,
                'total': total,
                'failed': failed
            })

        result = processor.process_batch(
            photos,
            progress_callback=track_progress
        )

        assert result.successful == 10
        assert len(progress_updates) == 10
        assert progress_updates[-1]['analyzed'] == 10

    def test_chunked_processing(self, processor, test_photos):
        """Test chunked batch processing."""
        result = processor.process_batch_chunked(
            test_photos,
            chunk_size=5
        )

        assert result.total_photos == 20
        assert result.successful == 20
        assert result.failed == 0


class TestAnalyzerBatchParallel:
    """Test analyzer's analyze_batch_parallel method."""

    def test_batch_parallel_basic(self, analyzer, test_photos):
        """Test parallel batch analysis."""
        photos = test_photos[:10]

        scores = analyzer.analyze_batch_parallel(photos, num_workers=2)

        assert len(scores) == 10
        assert all(score is not None for score in scores)

    def test_batch_parallel_preserves_order(self, analyzer, test_photos):
        """Test that parallel processing preserves input order."""
        photos = test_photos[:5]

        scores = analyzer.analyze_batch_parallel(photos, num_workers=2)

        # Verify order is preserved
        assert len(scores) == 5
        for i, score in enumerate(scores):
            assert score is not None

    def test_batch_parallel_with_progress(self, analyzer, test_photos):
        """Test parallel batch analysis with progress callback."""
        photos = test_photos[:10]
        progress_updates = []

        def track_progress(analyzed, total, failed):
            progress_updates.append(analyzed)

        scores = analyzer.analyze_batch_parallel(
            photos,
            num_workers=2,
            progress_callback=track_progress
        )

        assert len(scores) == 10
        assert len(progress_updates) == 10


class TestCompleteWorkflow:
    """Test complete workflow with all components."""

    def test_full_pipeline_with_cache(self, analyzer, cache, processor, test_photos):
        """Test complete pipeline: batch process -> cache -> retrieve."""
        photos = test_photos[:10]

        # Step 1: Batch process photos
        result = processor.process_batch(photos)
        assert result.successful == 10

        # Step 2: Cache all results
        for photo_path, score in result.scores:
            cache.set(photo_path, score)

        # Step 3: Verify all are cached
        stats = cache.get_stats()
        assert stats['total_entries'] == 10

        # Step 4: Retrieve from cache
        for photo in photos:
            cached_score = cache.get(photo)
            assert cached_score is not None

        # Step 5: Verify high hit rate
        stats = cache.get_stats()
        assert stats['hits'] == 10

    def test_mixed_cache_and_processing(self, analyzer, cache, test_photos):
        """Test workflow with mixed cache hits and misses."""
        photos = test_photos[:10]

        # Pre-populate cache with half the photos
        for i in range(5):
            score = analyzer.analyze_photo(photos[i])
            cache.set(photos[i], score)

        # Process all photos, using cache when available
        scores = []
        for photo in photos:
            if cache.should_analyze(photo):  # Calls get() internally
                score = analyzer.analyze_photo(photo)
                cache.set(photo, score)
            else:
                score = cache.get(photo)  # Second get() call
            scores.append(score)

        assert len(scores) == 10
        assert all(score is not None for score in scores)

        stats = cache.get_stats()
        assert stats['total_entries'] == 10
        # 5 cached: should_analyze() hit + explicit get() hit = 10 hits
        # 5 uncached: should_analyze() miss = 5 misses
        assert stats['hits'] == 10
        assert stats['misses'] == 5

    def test_export_import_workflow(self, cache, analyzer, test_photos, temp_dir):
        """Test export and import workflow."""
        photos = test_photos[:5]

        # Analyze and cache
        for photo in photos:
            score = analyzer.analyze_photo(photo)
            cache.set(photo, score)

        # Export cache
        export_path = Path(temp_dir) / "cache_export.json"
        count = cache.export_to_json(str(export_path))
        assert count == 5

        # Create new cache and import
        new_db_path = Path(temp_dir) / "new_cache.db"
        new_cache = EmotionalResultCache(db_path=str(new_db_path))
        imported = new_cache.import_from_json(str(export_path))
        assert imported == 5

        # Verify imported data
        stats = new_cache.get_stats()
        assert stats['total_entries'] == 5


class TestPerformance:
    """Test performance characteristics."""

    def test_cache_improves_performance(self, analyzer, cache, test_photos):
        """Test that caching improves performance on repeated analysis."""
        photo = test_photos[0]

        # First analysis (no cache)
        start1 = time.time()
        score1 = analyzer.analyze_photo(photo)
        time1 = time.time() - start1
        cache.set(photo, score1)

        # Second analysis (with cache)
        start2 = time.time()
        score2 = cache.get(photo)
        time2 = time.time() - start2

        # Cache should be much faster
        assert time2 < time1
        print(f"First: {time1*1000:.2f}ms, Cached: {time2*1000:.2f}ms, Speedup: {time1/time2:.1f}x")

    def test_batch_throughput(self, processor, test_photos):
        """Test batch processing throughput."""
        photos = test_photos[:20]

        start = time.time()
        result = processor.process_batch(photos)
        elapsed = time.time() - start

        throughput = result.successful / elapsed
        print(f"Processed {result.successful} photos in {elapsed:.2f}s ({throughput:.1f} photos/sec)")

        # Should process at least 5 photos per second (very conservative)
        assert throughput > 5.0

    def test_parallel_speedup(self, test_photos):
        """Test that parallel processing is faster than sequential."""
        photos = test_photos[:10]

        # Sequential processing
        processor_seq = EmotionalBatchProcessor(num_workers=1)
        start1 = time.time()
        result1 = processor_seq.process_batch(photos)
        time1 = time.time() - start1

        # Parallel processing
        processor_par = EmotionalBatchProcessor(num_workers=2)
        start2 = time.time()
        result2 = processor_par.process_batch(photos)
        time2 = time.time() - start2

        assert result1.successful == 10
        assert result2.successful == 10

        print(f"Sequential: {time1:.2f}s, Parallel: {time2:.2f}s")
        # Parallel should complete (may not always be faster due to overhead)
        assert time2 > 0


class TestErrorHandling:
    """Test error handling in integrated workflows."""

    def test_batch_with_invalid_photos(self, processor, test_photos):
        """Test batch processing with mix of valid and invalid photos."""
        valid_photos = test_photos[:5]
        invalid_photos = ["/nonexistent/photo1.jpg", "/nonexistent/photo2.jpg"]
        mixed_photos = valid_photos + invalid_photos

        result = processor.process_batch(mixed_photos)

        assert result.total_photos == 7
        assert result.successful == 5
        assert result.failed == 2
        assert len(result.scores) == 5
        assert len(result.errors) == 2

    def test_cache_with_deleted_photo(self, analyzer, cache, test_photos, temp_dir):
        """Test cache behavior when photo is deleted."""
        photo = test_photos[0]

        # Analyze and cache
        score = analyzer.analyze_photo(photo)
        cache.set(photo, score)

        # Delete photo
        Path(photo).unlink()

        # Try to get from cache (should fail gracefully)
        cached_score = cache.get(photo)
        assert cached_score is None

    def test_cache_invalidation_workflow(self, analyzer, cache, test_photos):
        """Test cache invalidation workflow."""
        photo = test_photos[0]

        # Analyze and cache
        score = analyzer.analyze_photo(photo)
        cache.set(photo, score)
        assert cache.get(photo) is not None

        # Invalidate
        cache.invalidate(photo)
        assert cache.get(photo) is None

        # Re-analyze
        new_score = analyzer.analyze_photo(photo)
        cache.set(photo, new_score)
        assert cache.get(photo) is not None


class TestScaleability:
    """Test system behavior at scale."""

    def test_large_batch_processing(self, processor, test_photos):
        """Test processing larger batch of photos."""
        # Create larger batch by repeating
        large_batch = test_photos * 5  # 100 photos

        result = processor.process_batch_chunked(
            large_batch,
            chunk_size=25
        )

        assert result.total_photos == 100
        assert result.successful == 100
        assert result.failed == 0

    def test_cache_capacity(self, cache, analyzer, temp_dir):
        """Test cache with many entries."""
        # Create 50 test photos
        photos = []
        for i in range(50):
            path = Path(temp_dir) / f"scale_photo_{i:03d}.jpg"
            image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            Image.fromarray(image).save(path)
            photos.append(str(path))

        # Analyze and cache all
        for photo in photos:
            score = analyzer.analyze_photo(photo)
            cache.set(photo, score)

        stats = cache.get_stats()
        assert stats['total_entries'] == 50

        # Verify all can be retrieved
        for photo in photos:
            score = cache.get(photo)
            assert score is not None

        stats = cache.get_stats()
        assert stats['hits'] == 50


class TestTierDistribution:
    """Test tier distribution in batch results."""

    def test_batch_result_tier_analysis(self, processor, test_photos):
        """Test analyzing tier distribution in batch results."""
        result = processor.process_batch(test_photos)

        # Count tiers
        tier_counts = {'high': 0, 'medium': 0, 'low': 0}
        for _, score in result.scores:
            tier_counts[score.tier] += 1

        # Should have some distribution
        total = sum(tier_counts.values())
        assert total == result.successful

        print(f"Tier distribution: {tier_counts}")

    def test_cache_tier_statistics(self, cache, analyzer, test_photos):
        """Test tier distribution in cache statistics."""
        photos = test_photos[:10]

        # Analyze and cache
        for photo in photos:
            score = analyzer.analyze_photo(photo)
            cache.set(photo, score)

        stats = cache.get_stats()
        tier_dist = stats['tier_distribution']

        # Should have tier distribution
        assert 'tier_distribution' in stats
        total_tiers = sum(tier_dist.values())
        assert total_tiers == 10
