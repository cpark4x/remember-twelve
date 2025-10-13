"""
EXIF utilities for extracting datetime information from photos.

Provides functions to extract photo timestamps from EXIF metadata,
with fallbacks to file modification time.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import os

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


def extract_datetime(photo_path: Path) -> Optional[datetime]:
    """
    Extract datetime from photo EXIF metadata.

    Tries multiple EXIF fields in order:
    1. DateTimeOriginal (when photo was taken)
    2. DateTime (when file was last modified in camera)
    3. DateTimeDigitized (when photo was digitized)

    Falls back to file modification time if no EXIF data.

    Args:
        photo_path: Path to the photo file

    Returns:
        datetime object, or None if extraction fails

    Examples:
        >>> dt = extract_datetime(Path('photo.jpg'))
        >>> print(dt.strftime('%Y-%m-%d'))
        2024-06-15
    """
    if not photo_path.exists():
        return None

    # Try EXIF extraction
    if PILLOW_AVAILABLE:
        exif_datetime = _extract_exif_datetime(photo_path)
        if exif_datetime:
            return exif_datetime

    # Fallback to file modification time
    return _get_file_mtime(photo_path)


def get_month(photo_path: Path) -> Optional[int]:
    """
    Get the month (1-12) when photo was taken.

    Args:
        photo_path: Path to the photo file

    Returns:
        Month as integer (1-12), or None if extraction fails

    Examples:
        >>> month = get_month(Path('photo.jpg'))
        >>> print(f"Month: {month}")
        Month: 6
    """
    dt = extract_datetime(photo_path)
    return dt.month if dt else None


def get_year(photo_path: Path) -> Optional[int]:
    """
    Get the year when photo was taken.

    Args:
        photo_path: Path to the photo file

    Returns:
        Year as integer, or None if extraction fails

    Examples:
        >>> year = get_year(Path('photo.jpg'))
        >>> print(f"Year: {year}")
        Year: 2024
    """
    dt = extract_datetime(photo_path)
    return dt.year if dt else None


def _extract_exif_datetime(photo_path: Path) -> Optional[datetime]:
    """
    Extract datetime from EXIF metadata using PIL.

    Args:
        photo_path: Path to the photo file

    Returns:
        datetime object, or None if extraction fails
    """
    try:
        with Image.open(photo_path) as img:
            exif_data = img._getexif()

            if not exif_data:
                return None

            # Try different EXIF datetime fields in order of preference
            datetime_fields = [
                36867,  # DateTimeOriginal
                306,    # DateTime
                36868,  # DateTimeDigitized
            ]

            for field_id in datetime_fields:
                if field_id in exif_data:
                    datetime_str = exif_data[field_id]
                    return _parse_exif_datetime(datetime_str)

            return None

    except (AttributeError, KeyError, OSError, IOError):
        return None


def _parse_exif_datetime(datetime_str: str) -> Optional[datetime]:
    """
    Parse EXIF datetime string to datetime object.

    EXIF datetime format: "YYYY:MM:DD HH:MM:SS"

    Args:
        datetime_str: EXIF datetime string

    Returns:
        datetime object, or None if parsing fails
    """
    try:
        # Try standard EXIF format: "2024:06:15 14:30:45"
        return datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
    except (ValueError, TypeError):
        pass

    try:
        # Try alternate format without time: "2024:06:15"
        return datetime.strptime(datetime_str, "%Y:%m:%d")
    except (ValueError, TypeError):
        pass

    return None


def _get_file_mtime(photo_path: Path) -> Optional[datetime]:
    """
    Get file modification time as fallback.

    Args:
        photo_path: Path to the photo file

    Returns:
        datetime object, or None if extraction fails
    """
    try:
        mtime = os.path.getmtime(photo_path)
        return datetime.fromtimestamp(mtime)
    except (OSError, ValueError):
        return None


def get_datetime_info(photo_path: Path) -> dict:
    """
    Get comprehensive datetime information about a photo.

    Returns both EXIF datetime and file modification time,
    along with metadata about which method was successful.

    Args:
        photo_path: Path to the photo file

    Returns:
        Dictionary with datetime info:
        {
            'datetime': datetime object (primary),
            'month': int (1-12),
            'year': int,
            'source': 'exif' or 'file_mtime',
            'exif_datetime': datetime or None,
            'file_mtime': datetime or None
        }

    Examples:
        >>> info = get_datetime_info(Path('photo.jpg'))
        >>> print(f"Taken in {info['month']}/{info['year']}")
        >>> print(f"Source: {info['source']}")
    """
    exif_dt = None
    if PILLOW_AVAILABLE:
        exif_dt = _extract_exif_datetime(photo_path)

    file_dt = _get_file_mtime(photo_path)

    # Prefer EXIF datetime
    primary_dt = exif_dt if exif_dt else file_dt
    source = 'exif' if exif_dt else 'file_mtime'

    return {
        'datetime': primary_dt,
        'month': primary_dt.month if primary_dt else None,
        'year': primary_dt.year if primary_dt else None,
        'source': source,
        'exif_datetime': exif_dt,
        'file_mtime': file_dt
    }


def is_from_year(photo_path: Path, year: int) -> bool:
    """
    Check if photo is from a specific year.

    Args:
        photo_path: Path to the photo file
        year: Year to check

    Returns:
        True if photo is from the specified year

    Examples:
        >>> if is_from_year(Path('photo.jpg'), 2024):
        ...     print("Photo from 2024")
    """
    photo_year = get_year(photo_path)
    return photo_year == year if photo_year else False


def filter_by_year(photo_paths: list[Path], year: int) -> list[Path]:
    """
    Filter photos by year.

    Args:
        photo_paths: List of photo paths
        year: Year to filter by

    Returns:
        List of photos from the specified year

    Examples:
        >>> photos_2024 = filter_by_year(all_photos, 2024)
        >>> print(f"Found {len(photos_2024)} photos from 2024")
    """
    return [p for p in photo_paths if is_from_year(p, year)]


def group_by_month(photo_paths: list[Path]) -> dict[int, list[Path]]:
    """
    Group photos by month.

    Args:
        photo_paths: List of photo paths

    Returns:
        Dictionary mapping month (1-12) to list of photo paths

    Examples:
        >>> by_month = group_by_month(photo_paths)
        >>> for month, photos in by_month.items():
        ...     print(f"Month {month}: {len(photos)} photos")
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for photo_path in photo_paths:
        month = get_month(photo_path)
        if month:
            grouped[month].append(photo_path)

    return dict(grouped)
