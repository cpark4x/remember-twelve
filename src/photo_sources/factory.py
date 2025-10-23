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

        Args:
            credentials_path: Path to OAuth credentials.json
            cache_dir: Optional cache directory

        Returns:
            GooglePhotosSource
        """
        from .google_photos_source import GooglePhotosSource
        return GooglePhotosSource(credentials_path, cache_dir=cache_dir)
