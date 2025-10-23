"""
PhotoSource Abstract Base Class

Defines the interface for all photo sources (local, Google Photos, iCloud, etc.)
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any
from datetime import datetime


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
