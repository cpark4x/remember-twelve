"""
PhotoSourceFactory - Create appropriate photo source based on configuration.

Simple factory pattern for creating LocalPhotoSource or GooglePhotosSource
based on user configuration.
"""

from typing import Dict, Any
from pathlib import Path

from .base import PhotoSource


class PhotoSourceFactory:
    """
    Factory for creating photo sources.

    Usage:
        >>> config = {'source_type': 'local', 'path': '/photos'}
        >>> source = PhotoSourceFactory.create(config)
        >>> for photo in source.scan(year=2024):
        ...     print(photo)
    """

    @staticmethod
    def create(config: Dict[str, Any]) -> PhotoSource:
        """
        Create appropriate PhotoSource from configuration.

        Args:
            config: Configuration dict with keys:
                source_type: 'local' | 'google_photos'

                For local:
                    path: str (directory path)

                For google_photos:
                    credentials_path: str (path to credentials.json)
                    cache_dir: str (optional, temp cache directory)

        Returns:
            PhotoSource instance

        Raises:
            ValueError: If unknown source type or missing required config

        Examples:
            # Local source
            >>> config = {
            ...     'source_type': 'local',
            ...     'path': '/Users/john/Photos'
            ... }
            >>> source = PhotoSourceFactory.create(config)

            # Google Photos source
            >>> config = {
            ...     'source_type': 'google_photos',
            ...     'credentials_path': '~/.remember_twelve/credentials.json'
            ... }
            >>> source = PhotoSourceFactory.create(config)
        """
        source_type = config.get('source_type', 'local')

        if source_type == 'local':
            from .local_photo_source import LocalPhotoSource

            path = config.get('path')
            if not path:
                raise ValueError("Local source requires 'path' in config")

            return LocalPhotoSource(path)

        elif source_type == 'google_photos':
            from .google_photos_source import GooglePhotosSource

            credentials_path = config.get('credentials_path')
            if not credentials_path:
                raise ValueError(
                    "Google Photos source requires 'credentials_path' in config"
                )

            cache_dir = config.get('cache_dir')
            return GooglePhotosSource(credentials_path, cache_dir=cache_dir)

        else:
            raise ValueError(f"Unknown source type: {source_type}")

    @staticmethod
    def create_local(path: str) -> PhotoSource:
        """
        Convenience method to create local source.

        Args:
            path: Directory path to scan

        Returns:
            LocalPhotoSource
        """
        from .local_photo_source import LocalPhotoSource
        return LocalPhotoSource(path)

    @staticmethod
    def create_google_photos(
        credentials_path: str,
        cache_dir: str = None
    ) -> PhotoSource:
        """
        Convenience method to create Google Photos source.

        ⚠️ DEPRECATED: Google Photos API no longer works (scope deprecated March 2025).
        Use create_local() or create_local_filesystem() instead with Google Takeout exports.
        See GOOGLE_PHOTOS_DEPRECATION.md for details.

        Args:
            credentials_path: Path to OAuth credentials.json
            cache_dir: Optional cache directory

        Returns:
            GooglePhotosSource (non-functional)
        """
        import warnings
        warnings.warn(
            "create_google_photos() is deprecated. Google Photos API no longer works. "
            "Use create_local() with Google Takeout exports instead. "
            "See GOOGLE_PHOTOS_DEPRECATION.md",
            DeprecationWarning,
            stacklevel=2
        )
        from .google_photos_source import GooglePhotosSource
        return GooglePhotosSource(credentials_path, cache_dir=cache_dir)

    @staticmethod
    def create_local_filesystem(path: str) -> PhotoSource:
        """
        Create a local filesystem photo source.

        Recommended for:
        - Google Takeout exports
        - Local photo folders
        - Google Drive sync folders
        - Any organized photo collection

        Args:
            path: Directory path containing photos

        Returns:
            LocalPhotoSource

        Example:
            >>> # Google Takeout
            >>> source = PhotoSourceFactory.create_local_filesystem(
            ...     '~/Downloads/Takeout/Google Photos'
            ... )
            >>>
            >>> # Local folder
            >>> source = PhotoSourceFactory.create_local_filesystem(
            ...     '~/Pictures/2023'
            ... )
        """
        from .local_photo_source import LocalPhotoSource
        return LocalPhotoSource(path)
