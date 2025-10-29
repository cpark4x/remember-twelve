"""
PhotoSource Abstract Base Class

Defines the interface for all photo sources (local, Google Photos, iCloud, etc.)
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class PhotoMetadata:
    """
    Lightweight photo metadata for pre-filtering (without download).

    This enables two-phase curation:
    - Phase 1: Pre-filter using metadata to ~50 candidates
    - Phase 2: Download and deeply analyze only candidates

    Attributes:
        id: Unique photo identifier
        timestamp: Photo creation time
        month: Month (1-12) or None
        width: Image width in pixels
        height: Image height in pixels
        file_size: File size in bytes
        mime_type: MIME type (image/jpeg, image/heic, etc.)
        source_url: Original URL in source (Google Photos, etc.)
    """
    id: str
    timestamp: Optional[datetime]
    month: Optional[int]
    width: Optional[int]
    height: Optional[int]
    file_size: int
    mime_type: str
    source_url: Optional[str]

    def metadata_score(self) -> float:
        """
        Calculate pre-filtering score based on metadata alone.

        Scoring factors:
        - Resolution: Higher resolution = better (normalized 0-100)
        - File size: Larger files often = better quality (normalized 0-100)
        - Format: HEIC/HEIF > JPG > PNG (0-100 scale)

        Returns:
            Score 0-100 based on metadata quality signals
        """
        score = 0.0

        # Resolution score (0-50 points)
        if self.width and self.height:
            megapixels = (self.width * self.height) / 1_000_000
            # Normalize: 1MP=0, 12MP+=50
            resolution_score = min(50.0, (megapixels / 12.0) * 50.0)
            score += resolution_score

        # File size score (0-30 points)
        # Larger files often indicate better quality
        # Normalize: 100KB=0, 5MB+=30
        size_mb = self.file_size / (1024 * 1024)
        size_score = min(30.0, (size_mb / 5.0) * 30.0)
        score += size_score

        # Format score (0-20 points)
        format_scores = {
            'image/heic': 20.0,
            'image/heif': 20.0,
            'image/jpeg': 15.0,
            'image/jpg': 15.0,
            'image/png': 10.0
        }
        score += format_scores.get(self.mime_type.lower(), 5.0)

        return score


class PhotoSource(ABC):
    """
    Abstract base class for photo sources.

    All photo sources must implement this interface to work with
    the curation pipeline. The interface is designed to be simple
    and flexible:

    - scan() returns an iterator of photo paths (local or temp)
    - get_metadata() provides photo metadata (timestamp, location, etc.)
    - get_original_url() links back to original source (Google Photos URL)
    - cleanup() removes temporary resources

    Design Philosophy (Ruthless Simplicity):
    - Return local file paths - analyzers don't care about source
    - Handle download/caching internally
    - Clean separation between source and analysis
    """

    @abstractmethod
    def scan(
        self,
        year: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Iterator[str]:
        """
        Scan for photos, optionally filtered by date.

        Args:
            year: Filter photos to this specific year (e.g., 2024)
            start_date: Filter photos after this date (inclusive)
            end_date: Filter photos before this date (inclusive)

        Returns:
            Iterator yielding absolute file paths (local or temp)

        Yields:
            str: Absolute path to photo file

        Raises:
            ValueError: If invalid date range provided

        Guarantees:
            - All paths are valid, readable files at time of yield
            - Files are images (jpg, jpeg, png, heic, heif)
            - Paths are absolute (not relative)

        Example:
            >>> source = GooglePhotosSource(credentials)
            >>> for photo_path in source.scan(year=2024):
            ...     print(photo_path)
            /tmp/remember_twelve_cache/photo_abc123.jpg
            /tmp/remember_twelve_cache/photo_def456.jpg
        """
        pass

    @abstractmethod
    def get_metadata(self, photo_path: str) -> Dict[str, Any]:
        """
        Get metadata for a photo.

        Args:
            photo_path: Path returned by scan()

        Returns:
            Dictionary with metadata:
                - timestamp (datetime | None): Photo creation time
                - month (int | None): Month (1-12)
                - location (dict | None): {'lat': float, 'lon': float}
                - format (str): Image format ('jpg', 'png', 'heic')
                - width (int | None): Image width in pixels
                - height (int | None): Image height in pixels
                - file_size (int): File size in bytes

        Raises:
            FileNotFoundError: If photo_path doesn't exist
            ValueError: If photo_path not from this source

        Example:
            >>> metadata = source.get_metadata('/tmp/cache/photo.jpg')
            >>> print(metadata['timestamp'])
            2024-06-15 14:30:00
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up temporary resources.

        For local sources: No-op
        For cloud sources: Delete cached files, clear temp storage

        Should be called after curation completes or on error.
        Idempotent - safe to call multiple times.

        Example:
            >>> source = GooglePhotosSource(credentials)
            >>> try:
            ...     for photo in source.scan(year=2024):
            ...         analyze(photo)
            ... finally:
            ...     source.cleanup()
        """
        pass

    @abstractmethod
    def get_original_url(self, photo_path: str) -> Optional[str]:
        """
        Get original source URL for a photo.

        Args:
            photo_path: Path returned by scan()

        Returns:
            URL string for cloud sources (Google Photos, etc.)
            None for local sources

        Used to link curated photos back to their original location
        in Google Photos web/mobile app.

        Example:
            >>> url = source.get_original_url('/tmp/cache/photo.jpg')
            >>> print(url)
            https://photos.google.com/photo/abc123
        """
        pass

    def list_metadata(
        self,
        year: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Iterator[PhotoMetadata]:
        """
        List photo metadata without downloading (two-phase curation).

        This is the new efficient method for cloud sources that:
        1. Fetches metadata only (no downloads)
        2. Enables pre-filtering before expensive downloads
        3. Supports two-phase curation workflow

        For local sources: Can return metadata from EXIF/filesystem
        For cloud sources: Fetch from API without downloading images

        Args:
            year: Filter photos to this specific year
            start_date: Filter photos after this date (inclusive)
            end_date: Filter photos before this date (inclusive)

        Yields:
            PhotoMetadata: Lightweight metadata for each photo

        Raises:
            ValueError: If invalid date range provided
            NotImplementedError: If source doesn't support metadata listing

        Example:
            >>> source = GooglePhotosSource(credentials)
            >>> for metadata in source.list_metadata(year=2024):
            ...     score = metadata.metadata_score()
            ...     print(f"{metadata.id}: {score}")
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support list_metadata(). "
            "Use scan() for legacy sources."
        )

    def download_photo(self, photo_id: str, destination: str) -> None:
        """
        Download a specific photo by ID (two-phase curation).

        This is the targeted download method used after pre-filtering:
        1. Pre-filter using list_metadata()
        2. Download only selected candidates using this method
        3. Perform deep analysis on downloaded files

        Args:
            photo_id: Photo ID from PhotoMetadata.id
            destination: Local path to save photo

        Raises:
            ValueError: If photo_id not found
            IOError: If download fails
            NotImplementedError: If source doesn't support targeted downloads

        Example:
            >>> source = GooglePhotosSource(credentials)
            >>> metadata_list = list(source.list_metadata(year=2024))
            >>> # Pre-filter to top 50
            >>> top_50 = sorted(metadata_list, key=lambda m: m.metadata_score(), reverse=True)[:50]
            >>> # Download only candidates
            >>> for metadata in top_50:
            ...     source.download_photo(metadata.id, f'/tmp/{metadata.id}.jpg')
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support download_photo(). "
            "Use scan() for legacy sources."
        )
