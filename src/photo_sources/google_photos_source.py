"""
GooglePhotosSource - Google Photos PhotoSource Implementation

⚠️ DEPRECATED: Google deprecated the photoslibrary.readonly scope on March 31, 2025.
This integration NO LONGER WORKS and will always return 403 errors.

Use local file sources instead:
- Google Takeout exports (see docs/GOOGLE_TAKEOUT_GUIDE.md)
- Local photo folders
- Google Drive sync

See GOOGLE_PHOTOS_DEPRECATION.md for details.

---

Legacy implementation (non-functional) kept for reference:
- OAuth authentication
- Photo listing by date
- Temporary download and caching
- Automatic cleanup

Phase 1: Authentication and basic photo listing (MVP)
"""

import warnings

import tempfile
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from datetime import datetime

from .base import PhotoSource, PhotoMetadata
from .google_photos_client import GooglePhotosClient
from .token_manager import TokenManager
from .exceptions import AuthenticationError


class GooglePhotosSource(PhotoSource):
    """
    Photo source implementation for Google Photos.

    Features:
    - OAuth 2.0 authentication
    - Fetch photos by date range
    - Download to temporary cache
    - Track photo metadata and URLs
    - Automatic cleanup

    Usage:
        >>> source = GooglePhotosSource('credentials.json')
        >>> source.authenticate()
        'user@gmail.com'
        >>> for photo in source.scan(year=2024):
        ...     print(photo)  # temp file path
    """

    def __init__(
        self,
        credentials_path: str,
        cache_dir: Optional[str] = None,
        token_manager: Optional[TokenManager] = None
    ):
        """
        Initialize Google Photos source.

        ⚠️ DEPRECATED: This source no longer works due to Google API deprecation.
        Use local file sources instead. See GOOGLE_PHOTOS_DEPRECATION.md.

        Args:
            credentials_path: Path to OAuth credentials.json
            cache_dir: Optional cache directory (uses temp dir if None)
            token_manager: Optional TokenManager instance
        """
        # Emit deprecation warning
        warnings.warn(
            "GooglePhotosSource is deprecated and non-functional. "
            "Google deprecated the photoslibrary.readonly scope on March 31, 2025. "
            "Use local file sources instead (Google Takeout, local folders, etc.). "
            "See GOOGLE_PHOTOS_DEPRECATION.md for details.",
            DeprecationWarning,
            stacklevel=2
        )

        self.credentials_path = credentials_path

        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._temp_cache = None
        else:
            # Use system temp directory
            self._temp_cache = tempfile.TemporaryDirectory(
                prefix='remember_twelve_'
            )
            self.cache_dir = Path(self._temp_cache.name)

        # Initialize client
        self.client = GooglePhotosClient(credentials_path, token_manager)

        # Track mapping: temp_path -> photo_metadata
        self.photo_map: Dict[str, Dict[str, Any]] = {}

        # Track downloaded files for cleanup
        self.downloaded_files: list[Path] = []

        # Track photo metadata by ID (for two-phase curation)
        self._photo_metadata_cache: Dict[str, Dict[str, Any]] = {}

    def authenticate(self, user_email: Optional[str] = None) -> str:
        """
        Authenticate with Google Photos.

        Args:
            user_email: Optional user email for token lookup

        Returns:
            Authenticated user's email

        Raises:
            AuthenticationError: If authentication fails
        """
        return self.client.authenticate(user_email)

    def scan(
        self,
        year: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Iterator[str]:
        """
        Scan Google Photos library for photos.

        Args:
            year: Filter to specific year (e.g., 2024)
            start_date: Filter photos after this date
            end_date: Filter photos before this date

        Yields:
            Absolute path to downloaded photo (in cache)

        Raises:
            AuthenticationError: If not authenticated
            ValueError: If no date filter provided

        Note: Photos are downloaded to temp cache as they're yielded.
              Call cleanup() after done to remove cached files.
        """
        # Ensure authenticated
        if not self.client.is_authenticated():
            raise AuthenticationError(
                "Not authenticated. Call authenticate() first."
            )

        # Determine date range
        if year:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31, 23, 59, 59)
        elif not (start_date and end_date):
            raise ValueError(
                "Must provide either 'year' or both 'start_date' and 'end_date'"
            )

        # Fetch photos from Google Photos
        for photo_metadata in self.client.list_photos(start_date, end_date):
            # Download photo to cache
            photo_id = photo_metadata['id']
            filename = photo_metadata.get('filename', f'photo_{photo_id}.jpg')
            base_url = photo_metadata.get('baseUrl')

            if not base_url:
                continue  # Skip photos without download URL

            # Generate local cache path
            cache_path = self.cache_dir / filename
            counter = 1
            while cache_path.exists():
                # Handle filename collisions
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                cache_path = self.cache_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            # Download photo
            try:
                download_url = self.client.get_download_url(photo_id, base_url)
                self.client.download_photo(download_url, cache_path)

                # Track mapping
                self.photo_map[str(cache_path)] = photo_metadata
                self.downloaded_files.append(cache_path)

                # Yield path
                yield str(cache_path)

            except Exception as e:
                # Log error but continue with other photos
                print(f"Warning: Failed to download {filename}: {e}")
                continue

    def get_metadata(self, photo_path: str) -> Dict[str, Any]:
        """
        Get metadata for a photo.

        Args:
            photo_path: Path returned by scan()

        Returns:
            Dict with metadata keys:
                - timestamp: datetime of photo creation
                - month: int (1-12)
                - format: str (jpg, png, heic)
                - width: int
                - height: int
                - file_size: int (bytes)

        Raises:
            ValueError: If photo_path not from this source
        """
        photo_metadata = self.photo_map.get(photo_path)
        if not photo_metadata:
            raise ValueError(f"Photo not from this source: {photo_path}")

        # Extract metadata from Google Photos response
        media_metadata = photo_metadata.get('mediaMetadata', {})
        creation_time = media_metadata.get('creationTime')

        # Parse timestamp
        timestamp = None
        month = None
        if creation_time:
            timestamp = datetime.fromisoformat(
                creation_time.replace('Z', '+00:00')
            )
            month = timestamp.month

        # Get dimensions
        width = int(media_metadata.get('width', 0))
        height = int(media_metadata.get('height', 0))

        # Get file size
        file_size = 0
        photo_file = Path(photo_path)
        if photo_file.exists():
            file_size = photo_file.stat().st_size

        # Determine format from mime type
        mime_type = photo_metadata.get('mimeType', 'image/jpeg')
        format_map = {
            'image/jpeg': 'jpg',
            'image/png': 'png',
            'image/heic': 'heic',
            'image/heif': 'heif'
        }
        photo_format = format_map.get(mime_type, 'jpg')

        return {
            'timestamp': timestamp,
            'month': month,
            'location': None,  # TODO: Extract location if available
            'format': photo_format,
            'width': width,
            'height': height,
            'file_size': file_size
        }

    def get_original_url(self, photo_path: str) -> Optional[str]:
        """
        Get Google Photos URL for a photo.

        Args:
            photo_path: Path returned by scan()

        Returns:
            Google Photos product URL (opens in web/app)
            None if photo not found

        Example:
            >>> url = source.get_original_url('/tmp/cache/photo.jpg')
            >>> print(url)
            https://photos.google.com/photo/ABC123
        """
        photo_metadata = self.photo_map.get(photo_path)
        if not photo_metadata:
            return None

        return photo_metadata.get('productUrl')

    def cleanup(self) -> None:
        """
        Clean up temporary cached files.

        Removes all downloaded photos from cache directory.
        Safe to call multiple times (idempotent).
        """
        # Delete downloaded files
        for file_path in self.downloaded_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete {file_path}: {e}")

        self.downloaded_files.clear()
        self.photo_map.clear()

        # Clean up temp directory if we created it
        if self._temp_cache:
            try:
                self._temp_cache.cleanup()
            except Exception as e:
                print(f"Warning: Failed to cleanup temp directory: {e}")

    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.cleanup()

    def list_metadata(
        self,
        year: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Iterator[PhotoMetadata]:
        """
        List photo metadata without downloading (two-phase curation).

        This is the efficient method that:
        1. Fetches metadata from Google Photos API
        2. No downloads - just metadata
        3. Enables pre-filtering before expensive downloads

        Args:
            year: Filter to specific year
            start_date: Filter photos after this date
            end_date: Filter photos before this date

        Yields:
            PhotoMetadata: Lightweight metadata for each photo

        Raises:
            AuthenticationError: If not authenticated
            ValueError: If no date filter provided
        """
        # Ensure authenticated
        if not self.client.is_authenticated():
            raise AuthenticationError(
                "Not authenticated. Call authenticate() first."
            )

        # Determine date range
        if year:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31, 23, 59, 59)
        elif not (start_date and end_date):
            raise ValueError(
                "Must provide either 'year' or both 'start_date' and 'end_date'"
            )

        # Fetch metadata from Google Photos (no downloads)
        for photo_metadata in self.client.list_photos(start_date, end_date):
            photo_id = photo_metadata['id']

            # Cache metadata for later download
            self._photo_metadata_cache[photo_id] = photo_metadata

            # Extract metadata fields
            media_metadata = photo_metadata.get('mediaMetadata', {})
            creation_time = media_metadata.get('creationTime')

            # Parse timestamp
            timestamp = None
            month = None
            if creation_time:
                timestamp = datetime.fromisoformat(
                    creation_time.replace('Z', '+00:00')
                )
                month = timestamp.month

            # Get dimensions
            width = int(media_metadata.get('width', 0)) or None
            height = int(media_metadata.get('height', 0)) or None

            # Get file size (estimate from dimensions if not available)
            # Google Photos API doesn't always provide file size
            # Estimate: assume 3 bytes per pixel for compressed JPEG
            if width and height:
                estimated_size = width * height * 3
            else:
                estimated_size = 1_000_000  # 1MB default

            # Get mime type
            mime_type = photo_metadata.get('mimeType', 'image/jpeg')

            # Get source URL
            source_url = photo_metadata.get('productUrl')

            # Create PhotoMetadata
            metadata = PhotoMetadata(
                id=photo_id,
                timestamp=timestamp,
                month=month,
                width=width,
                height=height,
                file_size=estimated_size,
                mime_type=mime_type,
                source_url=source_url
            )

            yield metadata

    def download_photo(self, photo_id: str, destination: str) -> None:
        """
        Download a specific photo by ID (two-phase curation).

        This is used after pre-filtering to download only selected candidates.

        Args:
            photo_id: Photo ID from PhotoMetadata.id
            destination: Local path to save photo

        Raises:
            ValueError: If photo_id not found
            IOError: If download fails
        """
        # Get cached metadata
        photo_metadata = self._photo_metadata_cache.get(photo_id)
        if not photo_metadata:
            raise ValueError(f"Photo ID not found: {photo_id}")

        base_url = photo_metadata.get('baseUrl')
        if not base_url:
            raise ValueError(f"No download URL for photo: {photo_id}")

        # Download photo
        try:
            download_url = self.client.get_download_url(photo_id, base_url)
            self.client.download_photo(download_url, destination)

            # Track downloaded file
            dest_path = Path(destination)
            self.downloaded_files.append(dest_path)
            self.photo_map[destination] = photo_metadata

        except Exception as e:
            raise IOError(f"Failed to download photo {photo_id}: {e}") from e
