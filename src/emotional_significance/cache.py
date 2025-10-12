"""
Result Cache for Emotional Significance Detector

Provides caching of emotional scores using SQLite backend with photo hash-based
identification. Prevents re-analyzing photos that haven't changed.

Key Features:
- SHA-256 hash-based photo identification
- SQLite backend for persistent storage
- Cache hit/miss tracking
- TTL/expiration support
- Thread-safe operations

Examples:
    >>> from emotional_significance import EmotionalResultCache
    >>> cache = EmotionalResultCache('emotional_scores.db')
    >>>
    >>> # Check if photo needs analysis
    >>> if cache.should_analyze('photo.jpg'):
    ...     score = analyzer.analyze_photo('photo.jpg')
    ...     cache.set('photo.jpg', score)
    >>> else:
    ...     score = cache.get('photo.jpg')
"""

import sqlite3
import hashlib
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager

from .data_classes import EmotionalScore
from .config import EmotionalConfig

logger = logging.getLogger(__name__)


class EmotionalResultCache:
    """
    Cache for emotional significance analysis results.

    Uses SQLite database with photo file hash as key to detect when
    photos have been modified and need re-analysis.

    Examples:
        >>> cache = EmotionalResultCache('emotional_scores.db')
        >>> if cache.should_analyze('photo.jpg'):
        ...     score = analyzer.analyze_photo('photo.jpg')
        ...     cache.set('photo.jpg', score)
        >>> else:
        ...     score = cache.get('photo.jpg')
        ...     print(f"Cached score: {score.composite}")
    """

    def __init__(
        self,
        db_path: str = "emotional_scores.db",
        config: Optional[EmotionalConfig] = None
    ):
        """
        Initialize result cache.

        Args:
            db_path: Path to SQLite database file
            config: Emotional analyzer configuration (uses default if None)
        """
        from .config import get_default_config

        self.db_path = db_path
        self.config = config or get_default_config()

        # Statistics
        self._hits = 0
        self._misses = 0

        # Initialize database
        self._init_database()

        logger.info(f"Emotional result cache initialized at {db_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database with schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emotional_scores (
                    photo_hash TEXT PRIMARY KEY,
                    photo_path TEXT NOT NULL,
                    face_count INTEGER NOT NULL,
                    face_coverage REAL NOT NULL,
                    emotion_score REAL NOT NULL,
                    intimacy_score REAL NOT NULL,
                    engagement_score REAL NOT NULL,
                    composite_score REAL NOT NULL,
                    tier TEXT NOT NULL,
                    metadata TEXT,
                    file_size INTEGER NOT NULL,
                    analyzed_at TIMESTAMP NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    CHECK (face_count >= 0),
                    CHECK (face_coverage BETWEEN 0 AND 1),
                    CHECK (emotion_score BETWEEN 0 AND 100),
                    CHECK (intimacy_score BETWEEN 0 AND 100),
                    CHECK (engagement_score BETWEEN 0 AND 100),
                    CHECK (composite_score BETWEEN 0 AND 100),
                    CHECK (tier IN ('high', 'medium', 'low'))
                )
            """)

            # Create indexes for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_photo_path
                ON emotional_scores(photo_path)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analyzed_at
                ON emotional_scores(analyzed_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tier
                ON emotional_scores(tier)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def _compute_file_hash(self, photo_path: str) -> tuple[str, int]:
        """
        Compute SHA-256 hash of photo file.

        Args:
            photo_path: Path to photo file

        Returns:
            Tuple of (hash_hex, file_size)

        Raises:
            FileNotFoundError: If photo doesn't exist
            IOError: If photo can't be read
        """
        path = Path(photo_path)

        if not path.exists():
            raise FileNotFoundError(f"Photo not found: {photo_path}")

        hasher = hashlib.sha256()
        file_size = 0

        # Read file in chunks for memory efficiency
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
                file_size += len(chunk)

        return hasher.hexdigest(), file_size

    def get(self, photo_path: str) -> Optional[EmotionalScore]:
        """
        Get cached emotional score for photo.

        Args:
            photo_path: Path to photo file

        Returns:
            EmotionalScore if found and valid, None otherwise

        Examples:
            >>> score = cache.get('photo.jpg')
            >>> if score:
            ...     print(f"Cached: {score.composite}")
            ... else:
            ...     print("Not cached")
        """
        try:
            # Compute current file hash
            current_hash, current_size = self._compute_file_hash(photo_path)

            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT photo_hash, face_count, face_coverage, emotion_score,
                           intimacy_score, engagement_score, composite_score,
                           tier, metadata, analyzed_at, algorithm_version
                    FROM emotional_scores
                    WHERE photo_hash = ?
                """, (current_hash,))

                row = cursor.fetchone()

                if row is None:
                    self._misses += 1
                    logger.debug(f"Cache miss: {photo_path}")
                    return None

                # Check if cache entry is expired (if TTL is enabled via config)
                # Note: EmotionalConfig doesn't have cache TTL by default,
                # but we support it if added
                if hasattr(self.config, 'cache_ttl_days') and self.config.cache_ttl_days > 0:
                    analyzed_at = datetime.fromisoformat(row['analyzed_at'])
                    ttl = timedelta(days=self.config.cache_ttl_days)

                    if datetime.now() - analyzed_at > ttl:
                        logger.debug(f"Cache expired: {photo_path}")
                        self._misses += 1
                        return None

                # Cache hit
                self._hits += 1
                logger.debug(f"Cache hit: {photo_path}")

                # Reconstruct metadata from JSON
                metadata = json.loads(row['metadata']) if row['metadata'] else {}

                return EmotionalScore(
                    face_count=row['face_count'],
                    face_coverage=row['face_coverage'],
                    emotion_score=row['emotion_score'],
                    intimacy_score=row['intimacy_score'],
                    engagement_score=row['engagement_score'],
                    composite=row['composite_score'],
                    tier=row['tier'],
                    metadata=metadata
                )

        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Error reading photo for cache lookup: {e}")
            return None

    def set(self, photo_path: str, score: EmotionalScore) -> bool:
        """
        Store emotional score in cache.

        Args:
            photo_path: Path to photo file
            score: Emotional score to cache

        Returns:
            True if successfully cached, False otherwise

        Examples:
            >>> score = analyzer.analyze_photo('photo.jpg')
            >>> cache.set('photo.jpg', score)
            True
        """
        try:
            # Compute file hash
            photo_hash, file_size = self._compute_file_hash(photo_path)

            # Serialize metadata to JSON
            metadata_json = json.dumps(score.metadata) if score.metadata else None

            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO emotional_scores (
                        photo_hash, photo_path, face_count, face_coverage,
                        emotion_score, intimacy_score, engagement_score,
                        composite_score, tier, metadata, file_size,
                        analyzed_at, algorithm_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    photo_hash,
                    photo_path,
                    score.face_count,
                    score.face_coverage,
                    score.emotion_score,
                    score.intimacy_score,
                    score.engagement_score,
                    score.composite,
                    score.tier,
                    metadata_json,
                    file_size,
                    datetime.now().isoformat(),
                    self.config.algorithm_version
                ))

                conn.commit()

            logger.debug(f"Cached score for: {photo_path}")
            return True

        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Error caching score: {e}")
            return False

    def should_analyze(self, photo_path: str) -> bool:
        """
        Check if photo should be analyzed (not in cache or changed).

        Args:
            photo_path: Path to photo file

        Returns:
            True if photo should be analyzed, False if cached score is valid

        Examples:
            >>> if cache.should_analyze('photo.jpg'):
            ...     score = analyzer.analyze_photo('photo.jpg')
            ...     cache.set('photo.jpg', score)
        """
        return self.get(photo_path) is None

    def invalidate(self, photo_path: str) -> bool:
        """
        Invalidate (remove) cached score for photo.

        Args:
            photo_path: Path to photo file

        Returns:
            True if entry was removed, False otherwise

        Examples:
            >>> cache.invalidate('edited_photo.jpg')
            True
            >>> # Photo will be re-analyzed next time
        """
        try:
            photo_hash, _ = self._compute_file_hash(photo_path)

            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM emotional_scores
                    WHERE photo_hash = ?
                """, (photo_hash,))

                conn.commit()

                deleted = cursor.rowcount > 0

            if deleted:
                logger.info(f"Invalidated cache for: {photo_path}")

            return deleted

        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Error invalidating cache: {e}")
            return False

    def clear(self) -> int:
        """
        Clear all cached scores.

        Returns:
            Number of entries removed

        Examples:
            >>> removed = cache.clear()
            >>> print(f"Cleared {removed} cached scores")
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM emotional_scores")
            conn.commit()
            count = cursor.rowcount

        # Reset statistics
        self._hits = 0
        self._misses = 0

        logger.info(f"Cleared cache: {count} entries removed")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics

        Examples:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
            >>> print(f"Total entries: {stats['total_entries']}")
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM emotional_scores")
            total_entries = cursor.fetchone()['count']

            # Get tier distribution
            cursor = conn.execute("""
                SELECT tier, COUNT(*) as count
                FROM emotional_scores
                GROUP BY tier
            """)
            tier_distribution = {row['tier']: row['count'] for row in cursor}

        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

        return {
            'total_entries': total_entries,
            'hits': self._hits,
            'misses': self._misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'tier_distribution': tier_distribution
        }

    def get_all_scores(self) -> Dict[str, EmotionalScore]:
        """
        Get all cached scores.

        Returns:
            Dictionary mapping photo_path to EmotionalScore

        Examples:
            >>> all_scores = cache.get_all_scores()
            >>> for path, score in all_scores.items():
            ...     print(f"{path}: {score.composite}")
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT photo_path, face_count, face_coverage, emotion_score,
                       intimacy_score, engagement_score, composite_score,
                       tier, metadata
                FROM emotional_scores
            """)

            scores = {}
            for row in cursor:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}

                scores[row['photo_path']] = EmotionalScore(
                    face_count=row['face_count'],
                    face_coverage=row['face_coverage'],
                    emotion_score=row['emotion_score'],
                    intimacy_score=row['intimacy_score'],
                    engagement_score=row['engagement_score'],
                    composite=row['composite_score'],
                    tier=row['tier'],
                    metadata=metadata
                )

        return scores

    def export_to_json(self, output_path: str) -> int:
        """
        Export all cached scores to JSON file.

        Args:
            output_path: Path to output JSON file

        Returns:
            Number of entries exported

        Examples:
            >>> cache.export_to_json('emotional_scores_backup.json')
            1234
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT photo_path, photo_hash, face_count, face_coverage,
                       emotion_score, intimacy_score, engagement_score,
                       composite_score, tier, metadata, analyzed_at,
                       algorithm_version
                FROM emotional_scores
            """)

            entries = []
            for row in cursor:
                entries.append({
                    'photo_path': row['photo_path'],
                    'photo_hash': row['photo_hash'],
                    'face_count': row['face_count'],
                    'face_coverage': row['face_coverage'],
                    'emotion_score': row['emotion_score'],
                    'intimacy_score': row['intimacy_score'],
                    'engagement_score': row['engagement_score'],
                    'composite': row['composite_score'],
                    'tier': row['tier'],
                    'metadata': row['metadata'],
                    'analyzed_at': row['analyzed_at'],
                    'algorithm_version': row['algorithm_version']
                })

        with open(output_path, 'w') as f:
            json.dump(entries, f, indent=2)

        logger.info(f"Exported {len(entries)} cache entries to {output_path}")
        return len(entries)

    def import_from_json(self, input_path: str) -> int:
        """
        Import cached scores from JSON file.

        Args:
            input_path: Path to input JSON file

        Returns:
            Number of entries imported

        Examples:
            >>> cache.import_from_json('emotional_scores_backup.json')
            1234
        """
        with open(input_path, 'r') as f:
            entries = json.load(f)

        count = 0
        with self._get_connection() as conn:
            for entry in entries:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO emotional_scores (
                            photo_hash, photo_path, face_count, face_coverage,
                            emotion_score, intimacy_score, engagement_score,
                            composite_score, tier, metadata, file_size,
                            analyzed_at, algorithm_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry['photo_hash'],
                        entry['photo_path'],
                        entry['face_count'],
                        entry['face_coverage'],
                        entry['emotion_score'],
                        entry['intimacy_score'],
                        entry['engagement_score'],
                        entry['composite'],
                        entry['tier'],
                        entry.get('metadata'),
                        0,  # file_size not in export
                        entry['analyzed_at'],
                        entry.get('algorithm_version', 'v1.0')
                    ))
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to import entry: {e}")

            conn.commit()

        logger.info(f"Imported {count} cache entries from {input_path}")
        return count
