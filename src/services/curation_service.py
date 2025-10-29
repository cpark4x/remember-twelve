"""
CurationService - End-to-end photo curation orchestration.

Orchestrates the complete curation pipeline:
1. Authenticate with photo source (Google Photos)
2. Run curation algorithm (TwelveCurator)
3. Save results to database

This module extracts business logic from CLI scripts to make it
reusable across API, CLI, and other interfaces.
"""

import sqlite3
import json
from typing import Optional, Callable
from pathlib import Path

from src.photo_sources.factory import PhotoSourceFactory
from src.twelve_curator.curator import TwelveCurator
from src.twelve_curator.data_classes import TwelveSelection, CurationConfig
from src.database import photo_repository, curation_repository


class AuthenticationError(Exception):
    """Raised when Google Photos authentication fails"""
    pass


class CurationService:
    """
    Service layer for photo curation operations.

    Orchestrates the complete curation pipeline from authentication
    through to database persistence.

    Examples:
        >>> service = CurationService()
        >>>
        >>> # Curate from Google Photos
        >>> selection = service.curate_year(
        ...     year=2024,
        ...     strategy='balanced',
        ...     credentials_path='~/.remember_twelve/credentials.json'
        ... )
        >>>
        >>> # Save to database
        >>> import sqlite3
        >>> conn = sqlite3.connect('remember_twelve.db')
        >>> curation_id = service.save_to_database(conn, selection)
        >>> conn.commit()
    """

    def curate_year(
        self,
        year: int,
        strategy: str,
        credentials_path: str,
        progress_callback: Optional[Callable[[int, Optional[int], str], None]] = None
    ) -> TwelveSelection:
        """
        Curate best 12 photos from Google Photos for a given year.

        Complete pipeline:
        1. Create Google Photos source
        2. Authenticate
        3. Create curator with strategy
        4. Run curation algorithm
        5. Return TwelveSelection

        Args:
            year: Year to curate (e.g., 2024)
            strategy: Curation strategy ('balanced', 'aesthetic_first',
                     'people_first', 'top_heavy')
            credentials_path: Path to Google Photos OAuth credentials.json
            progress_callback: Optional callback(current, total, message)
                             Called during photo analysis

        Returns:
            TwelveSelection containing best 12 photos with metadata

        Raises:
            AuthenticationError: If Google Photos authentication fails
            ValueError: If invalid strategy or year
            Exception: Other curation failures

        Examples:
            >>> service = CurationService()
            >>>
            >>> # Basic usage
            >>> selection = service.curate_year(
            ...     year=2024,
            ...     strategy='balanced',
            ...     credentials_path='creds.json'
            ... )
            >>> print(f"Selected {len(selection.photos)} photos")
            >>>
            >>> # With progress tracking
            >>> def on_progress(current, total, msg):
            ...     print(f"[{current}] {msg}")
            >>>
            >>> selection = service.curate_year(
            ...     year=2024,
            ...     strategy='people_first',
            ...     credentials_path='creds.json',
            ...     progress_callback=on_progress
            ... )
        """
        if year < 1900 or year > 2100:
            raise ValueError(f"Invalid year: {year}")

        valid_strategies = ['balanced', 'aesthetic_first', 'people_first', 'top_heavy']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")

        try:
            source = PhotoSourceFactory.create_google_photos(str(credentials_path))
        except Exception as e:
            raise AuthenticationError(f"Failed to create Google Photos source: {e}") from e

        try:
            user_email = source.authenticate()
        except Exception as e:
            raise AuthenticationError(f"Google Photos authentication failed: {e}") from e

        if strategy == 'balanced':
            config = CurationConfig.balanced()
        elif strategy == 'aesthetic_first':
            config = CurationConfig.aesthetic_first()
        elif strategy == 'people_first':
            config = CurationConfig.people_first()
        elif strategy == 'top_heavy':
            config = CurationConfig.top_heavy()

        curator = TwelveCurator(config)

        try:
            # Use v2 method for efficient two-phase curation
            selection = curator.curate_from_source_v2(
                source,
                year=year,
                strategy=strategy,
                progress_callback=progress_callback
            )
            return selection
        except Exception as e:
            raise Exception(f"Curation failed: {e}") from e

    def save_to_database(
        self,
        conn: sqlite3.Connection,
        selection: TwelveSelection
    ) -> int:
        """
        Save curation results to database.

        Database operations:
        1. Insert/update photos with scores
        2. Create curation record
        3. Link photos to curation via curation_photos

        Note: Does NOT commit transaction - caller controls commit/rollback

        Args:
            conn: SQLite database connection
            selection: TwelveSelection from curate_year()

        Returns:
            curation_id: ID of created curation record

        Raises:
            ValueError: If selection is invalid or empty
            Exception: Database operation failures

        Examples:
            >>> import sqlite3
            >>> from src.services import CurationService
            >>>
            >>> service = CurationService()
            >>> selection = service.curate_year(2024, 'balanced', 'creds.json')
            >>>
            >>> conn = sqlite3.connect('remember_twelve.db')
            >>> try:
            ...     curation_id = service.save_to_database(conn, selection)
            ...     conn.commit()
            ...     print(f"Saved curation {curation_id}")
            ... except Exception as e:
            ...     conn.rollback()
            ...     raise
            ... finally:
            ...     conn.close()
        """
        if not selection.photos:
            raise ValueError("Cannot save empty selection")

        try:
            photo_data_list = []
            for photo in selection.photos:
                metadata = photo.metadata.copy() if photo.metadata else {}

                photo_data = {
                    'filename': photo.photo_path.name,
                    'source_path': str(photo.photo_path),
                    'captured_at': photo.timestamp.isoformat() if photo.timestamp else None,
                    'month': photo.month,
                    'year': selection.year,
                    'quality_score': photo.quality_score,
                    'emotional_score': photo.emotional_score,
                    'combined_score': photo.combined_score,
                    'metadata_json': json.dumps(metadata)
                }
                photo_data_list.append(photo_data)

            photo_ids = photo_repository.insert_photos_batch(conn, photo_data_list)

            curation_id = curation_repository.create_curation(
                conn,
                year=selection.year,
                strategy=selection.strategy,
                stats=selection.stats
            )

            photo_assignments = []
            for idx, photo_id in enumerate(photo_ids):
                month_slot = selection.photos[idx].month if selection.photos[idx].month else (idx + 1)
                photo_assignments.append((photo_id, month_slot))

            curation_repository.add_curation_photos_batch(
                conn,
                curation_id,
                photo_assignments
            )

            return curation_id

        except Exception as e:
            raise Exception(f"Failed to save curation to database: {e}") from e
