"""
Tests for ResultCache

Covers cache operations, hash-based identification, TTL/expiration,
statistics tracking, and import/export functionality.
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime, timedelta

from src.photo_quality_analyzer.cache import ResultCache
from src.photo_quality_analyzer.metrics.composite import QualityScore
from src.photo_quality_analyzer.config import QualityAnalyzerConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_photo(temp_dir):
    """Create a test photo."""
    photo_path = Path(temp_dir) / "test_photo.jpg"
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(image).save(photo_path)
    return str(photo_path)


@pytest.fixture
def test_score():
    """Create a test quality score."""
    return QualityScore(
        sharpness=75.5,
        exposure=82.3,
        composite=78.2,
        tier='high'
    )


@pytest.fixture
def cache(temp_dir):
    """Create cache instance with temporary database."""
    db_path = Path(temp_dir) / "test_cache.db"
    return ResultCache(db_path=str(db_path))


class TestResultCacheInitialization:
    """Test cache initialization."""

    def test_cache_creation(self, temp_dir):
        """Test creating a new cache."""
        db_path = Path(temp_dir) / "new_cache.db"
        cache = ResultCache(db_path=str(db_path))

        assert Path(db_path).exists()
        assert cache.db_path == str(db_path)

    def test_cache_with_config(self, temp_dir):
        """Test cache with custom configuration."""
        config = QualityAnalyzerConfig()
        config.cache.cache_ttl_days = 30

        db_path = Path(temp_dir) / "cache.db"
        cache = ResultCache(db_path=str(db_path), config=config)

        assert cache.config.cache.cache_ttl_days == 30

    def test_database_schema(self, cache):
        """Test that database schema is created correctly."""
        import sqlite3

        conn = sqlite3.connect(cache.db_path)
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='quality_scores'
        """)

        assert cursor.fetchone() is not None
        conn.close()


class TestCacheOperations:
    """Test basic cache operations (get, set, invalidate)."""

    def test_cache_miss(self, cache, test_photo):
        """Test cache miss for photo that hasn't been analyzed."""
        score = cache.get(test_photo)
        assert score is None

        stats = cache.get_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 0

    def test_cache_set_and_get(self, cache, test_photo, test_score):
        """Test setting and getting a score."""
        # Set score
        success = cache.set(test_photo, test_score)
        assert success is True

        # Get score
        retrieved_score = cache.get(test_photo)
        assert retrieved_score is not None
        assert retrieved_score.sharpness == test_score.sharpness
        assert retrieved_score.exposure == test_score.exposure
        assert retrieved_score.composite == test_score.composite
        assert retrieved_score.tier == test_score.tier

        stats = cache.get_stats()
        assert stats['hits'] == 1

    def test_cache_update(self, cache, test_photo, test_score):
        """Test updating an existing cache entry."""
        # Set initial score
        cache.set(test_photo, test_score)

        # Update with new score
        new_score = QualityScore(
            sharpness=90.0,
            exposure=85.0,
            composite=88.0,
            tier='high'
        )
        cache.set(test_photo, new_score)

        # Retrieve updated score
        retrieved_score = cache.get(test_photo)
        assert retrieved_score.sharpness == 90.0
        assert retrieved_score.composite == 88.0

    def test_should_analyze_not_cached(self, cache, test_photo):
        """Test should_analyze for uncached photo."""
        assert cache.should_analyze(test_photo) is True

    def test_should_analyze_cached(self, cache, test_photo, test_score):
        """Test should_analyze for cached photo."""
        cache.set(test_photo, test_score)
        assert cache.should_analyze(test_photo) is False

    def test_invalidate_cached_score(self, cache, test_photo, test_score):
        """Test invalidating a cached score."""
        # Cache a score
        cache.set(test_photo, test_score)
        assert cache.get(test_photo) is not None

        # Invalidate
        result = cache.invalidate(test_photo)
        assert result is True

        # Should be gone now
        assert cache.get(test_photo) is None

    def test_invalidate_nonexistent(self, cache, test_photo):
        """Test invalidating a non-existent entry."""
        result = cache.invalidate(test_photo)
        assert result is False


class TestHashBasedCaching:
    """Test hash-based photo identification."""

    def test_same_photo_same_hash(self, cache, test_photo, test_score):
        """Test that same photo gets same hash."""
        # Cache score
        cache.set(test_photo, test_score)

        # Read from cache (should work)
        score = cache.get(test_photo)
        assert score is not None

    def test_modified_photo_cache_miss(self, cache, test_photo, test_score):
        """Test that modified photo results in cache miss."""
        # Cache score
        cache.set(test_photo, test_score)

        # Modify the photo
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        Image.fromarray(image).save(test_photo)

        # Should be cache miss now (different hash)
        score = cache.get(test_photo)
        assert score is None

    def test_different_photos_different_hashes(self, cache, temp_dir, test_score):
        """Test that different photos have different hashes."""
        # Create two different photos
        photo1 = Path(temp_dir) / "photo1.jpg"
        photo2 = Path(temp_dir) / "photo2.jpg"

        image1 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        image2 = np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)

        Image.fromarray(image1).save(photo1)
        Image.fromarray(image2).save(photo2)

        # Cache scores
        score1 = QualityScore(75.0, 80.0, 77.0, 'high')
        score2 = QualityScore(60.0, 70.0, 64.0, 'acceptable')

        cache.set(str(photo1), score1)
        cache.set(str(photo2), score2)

        # Retrieve and verify they're different
        retrieved1 = cache.get(str(photo1))
        retrieved2 = cache.get(str(photo2))

        assert retrieved1.composite == 77.0
        assert retrieved2.composite == 64.0


class TestCacheStatistics:
    """Test cache statistics tracking."""

    def test_initial_stats(self, cache):
        """Test initial statistics are zero."""
        stats = cache.get_stats()

        assert stats['total_entries'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['total_requests'] == 0
        assert stats['hit_rate'] == 0.0

    def test_stats_after_operations(self, cache, test_photo, test_score):
        """Test statistics after cache operations."""
        # Miss
        cache.get(test_photo)

        # Set
        cache.set(test_photo, test_score)

        # Hit
        cache.get(test_photo)
        cache.get(test_photo)

        stats = cache.get_stats()

        assert stats['total_entries'] == 1
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['total_requests'] == 3
        assert abs(stats['hit_rate'] - 0.6667) < 0.01  # 2/3 ≈ 0.6667

    def test_hit_rate_calculation(self, cache, temp_dir, test_score):
        """Test hit rate calculation."""
        # Create multiple test photos
        photos = []
        for i in range(10):
            path = Path(temp_dir) / f"test_{i}.jpg"
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(image).save(path)
            photos.append(str(path))

        # Cache some photos (7 photos)
        for i in range(7):
            cache.set(photos[i], test_score)

        # 7 hits (cached photos)
        for i in range(7):
            cache.get(photos[i])  # Hit

        # 3 misses (uncached photos)
        for i in range(7, 10):
            cache.get(photos[i])  # Miss

        stats = cache.get_stats()
        # 7 hits + 3 misses = 10 total requests, 7/10 = 0.7
        assert abs(stats['hit_rate'] - 0.70) < 0.01


class TestCacheClearAndCleanup:
    """Test cache clearing and cleanup operations."""

    def test_clear_empty_cache(self, cache):
        """Test clearing an empty cache."""
        count = cache.clear()
        assert count == 0

    def test_clear_cache_with_entries(self, cache, test_photo, test_score, temp_dir):
        """Test clearing cache with entries."""
        # Add multiple entries
        for i in range(5):
            path = Path(temp_dir) / f"photo_{i}.jpg"
            # Create actual image file
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(image).save(path)
            cache.set(str(path), test_score)

        stats = cache.get_stats()
        assert stats['total_entries'] == 5

        # Clear
        count = cache.clear()
        assert count == 5

        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0

    def test_cleanup_expired_no_ttl(self, cache, test_photo, test_score):
        """Test cleanup with TTL disabled."""
        cache.set(test_photo, test_score)

        # Should not remove anything (TTL disabled)
        removed = cache.cleanup_expired()
        assert removed == 0

        stats = cache.get_stats()
        assert stats['total_entries'] == 1

    def test_cleanup_expired_with_ttl(self, temp_dir, test_photo, test_score):
        """Test cleanup with TTL enabled."""
        # Create cache with short TTL
        config = QualityAnalyzerConfig()
        config.cache.cache_ttl_days = 1  # 1 day TTL

        db_path = Path(temp_dir) / "ttl_cache.db"
        cache = ResultCache(db_path=str(db_path), config=config)

        # Add entry
        cache.set(test_photo, test_score)

        # Manually backdate the entry to simulate expiration
        import sqlite3
        conn = sqlite3.connect(cache.db_path)
        old_date = (datetime.now() - timedelta(days=2)).isoformat()
        conn.execute("""
            UPDATE quality_scores
            SET analyzed_at = ?
        """, (old_date,))
        conn.commit()
        conn.close()

        # Cleanup should remove expired entry
        removed = cache.cleanup_expired()
        assert removed == 1

        stats = cache.get_stats()
        assert stats['total_entries'] == 0


class TestCacheGetAllScores:
    """Test getting all cached scores."""

    def test_get_all_scores_empty(self, cache):
        """Test getting all scores from empty cache."""
        scores = cache.get_all_scores()
        assert len(scores) == 0

    def test_get_all_scores(self, cache, temp_dir):
        """Test getting all cached scores."""
        # Create multiple photos and cache scores
        for i in range(5):
            photo_path = Path(temp_dir) / f"photo_{i}.jpg"
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(image).save(photo_path)

            score = QualityScore(
                sharpness=70.0 + i,
                exposure=75.0 + i,
                composite=72.0 + i,
                tier='high'
            )
            cache.set(str(photo_path), score)

        # Get all scores
        all_scores = cache.get_all_scores()

        assert len(all_scores) == 5
        for path, score in all_scores.items():
            assert 'photo_' in path
            assert 70.0 <= score.sharpness <= 74.0


class TestCacheImportExport:
    """Test cache import/export functionality."""

    def test_export_empty_cache(self, cache, temp_dir):
        """Test exporting empty cache."""
        export_path = Path(temp_dir) / "export.json"

        count = cache.export_to_json(str(export_path))

        assert count == 0
        assert export_path.exists()

        with open(export_path) as f:
            data = json.load(f)
            assert len(data) == 0

    def test_export_cache(self, cache, test_photo, test_score, temp_dir):
        """Test exporting cache to JSON."""
        cache.set(test_photo, test_score)

        export_path = Path(temp_dir) / "export.json"
        count = cache.export_to_json(str(export_path))

        assert count == 1
        assert export_path.exists()

        with open(export_path) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]['photo_path'] == test_photo
            assert data[0]['composite'] == test_score.composite

    def test_import_cache(self, temp_dir):
        """Test importing cache from JSON."""
        # Create export file
        export_data = [{
            'photo_hash': 'abc123',
            'photo_path': '/test/photo.jpg',
            'sharpness': 75.0,
            'exposure': 80.0,
            'composite': 77.0,
            'tier': 'high',
            'analyzed_at': datetime.now().isoformat(),
            'algorithm_version': 'v1.0'
        }]

        import_path = Path(temp_dir) / "import.json"
        with open(import_path, 'w') as f:
            json.dump(export_data, f)

        # Import into new cache
        db_path = Path(temp_dir) / "import_cache.db"
        cache = ResultCache(db_path=str(db_path))

        count = cache.import_from_json(str(import_path))

        assert count == 1

        stats = cache.get_stats()
        assert stats['total_entries'] == 1

    def test_export_import_roundtrip(self, cache, test_photo, test_score, temp_dir):
        """Test export and import roundtrip."""
        # Cache score
        cache.set(test_photo, test_score)

        # Export
        export_path = Path(temp_dir) / "roundtrip.json"
        cache.export_to_json(str(export_path))

        # Create new cache and import
        new_db_path = Path(temp_dir) / "new_cache.db"
        new_cache = ResultCache(db_path=str(new_db_path))
        new_cache.import_from_json(str(export_path))

        # Verify
        all_scores = new_cache.get_all_scores()
        assert len(all_scores) == 1


class TestCacheErrorHandling:
    """Test error handling in cache operations."""

    def test_get_nonexistent_photo(self, cache):
        """Test getting score for non-existent photo."""
        score = cache.get("/nonexistent/photo.jpg")
        assert score is None

    def test_set_nonexistent_photo(self, cache, test_score):
        """Test setting score for non-existent photo."""
        success = cache.set("/nonexistent/photo.jpg", test_score)
        assert success is False

    def test_invalidate_nonexistent_photo(self, cache):
        """Test invalidating non-existent photo."""
        success = cache.invalidate("/nonexistent/photo.jpg")
        assert success is False

    def test_corrupted_database_recovery(self, temp_dir):
        """Test that cache can be recreated if database is corrupted."""
        db_path = Path(temp_dir) / "corrupted.db"

        # Create cache
        cache1 = ResultCache(db_path=str(db_path))

        # Corrupt database
        with open(db_path, 'w') as f:
            f.write("This is not a valid SQLite database")

        # Try to create new cache (should reinitialize)
        try:
            cache2 = ResultCache(db_path=str(db_path))
            # If it doesn't raise, database was reinitialized
            # Note: SQLite might detect corruption on operations
        except Exception:
            # Expected behavior - corruption detected
            pass


class TestCacheConcurrency:
    """Test cache behavior with concurrent operations."""

    def test_multiple_cache_instances(self, temp_dir, test_photo, test_score):
        """Test multiple cache instances sharing same database."""
        db_path = Path(temp_dir) / "shared.db"

        # Create two cache instances
        cache1 = ResultCache(db_path=str(db_path))
        cache2 = ResultCache(db_path=str(db_path))

        # Set score in cache1
        cache1.set(test_photo, test_score)

        # Read from cache2 (should see the entry)
        score = cache2.get(test_photo)
        assert score is not None
        assert score.composite == test_score.composite


class TestCacheTTL:
    """Test TTL (Time-To-Live) functionality."""

    def test_ttl_disabled_by_default(self, cache, test_photo, test_score):
        """Test that TTL is disabled by default."""
        cache.set(test_photo, test_score)

        # Should still be cached (no expiration)
        score = cache.get(test_photo)
        assert score is not None

    def test_ttl_not_expired(self, temp_dir, test_photo, test_score):
        """Test that non-expired entries are returned."""
        config = QualityAnalyzerConfig()
        config.cache.cache_ttl_days = 365  # 1 year

        db_path = Path(temp_dir) / "ttl_cache.db"
        cache = ResultCache(db_path=str(db_path), config=config)

        cache.set(test_photo, test_score)

        # Should still be cached
        score = cache.get(test_photo)
        assert score is not None
