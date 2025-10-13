"""
Tests for EXIF utils.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
from PIL import Image
import piexif

from src.twelve_curator.exif_utils import (
    extract_datetime,
    get_month,
    get_year,
    get_datetime_info,
    is_from_year,
    filter_by_year,
    group_by_month
)


@pytest.fixture
def temp_photo_with_exif():
    """Create a temporary photo with EXIF data."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a simple image
        img = Image.new('RGB', (100, 100), color='red')

        # Add EXIF data
        exif_dict = {
            "0th": {},
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: b"2024:06:15 14:30:45"
            }
        }
        exif_bytes = piexif.dump(exif_dict)

        img.save(f.name, exif=exif_bytes)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_photo_no_exif():
    """Create a temporary photo without EXIF data."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(f.name)
        yield Path(f.name)

    Path(f.name).unlink(missing_ok=True)


class TestExtractDatetime:
    """Tests for extract_datetime function."""

    def test_extract_from_exif(self, temp_photo_with_exif):
        """Should extract datetime from EXIF."""
        dt = extract_datetime(temp_photo_with_exif)

        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30

    def test_fallback_to_file_mtime(self, temp_photo_no_exif):
        """Should fall back to file modification time."""
        dt = extract_datetime(temp_photo_no_exif)

        assert dt is not None
        # Should be recent (file was just created)
        assert dt.year >= 2024

    def test_nonexistent_file(self):
        """Should return None for nonexistent file."""
        dt = extract_datetime(Path("nonexistent.jpg"))
        assert dt is None


class TestGetMonth:
    """Tests for get_month function."""

    def test_get_month_from_exif(self, temp_photo_with_exif):
        """Should get month from EXIF."""
        month = get_month(temp_photo_with_exif)

        assert month == 6

    def test_get_month_from_file(self, temp_photo_no_exif):
        """Should get month from file mtime."""
        month = get_month(temp_photo_no_exif)

        assert month is not None
        assert 1 <= month <= 12

    def test_nonexistent_file(self):
        """Should return None for nonexistent file."""
        month = get_month(Path("nonexistent.jpg"))
        assert month is None


class TestGetYear:
    """Tests for get_year function."""

    def test_get_year_from_exif(self, temp_photo_with_exif):
        """Should get year from EXIF."""
        year = get_year(temp_photo_with_exif)

        assert year == 2024

    def test_get_year_from_file(self, temp_photo_no_exif):
        """Should get year from file mtime."""
        year = get_year(temp_photo_no_exif)

        assert year is not None
        assert year >= 2024


class TestGetDatetimeInfo:
    """Tests for get_datetime_info function."""

    def test_comprehensive_info(self, temp_photo_with_exif):
        """Should get comprehensive datetime info."""
        info = get_datetime_info(temp_photo_with_exif)

        assert info['datetime'] is not None
        assert info['month'] == 6
        assert info['year'] == 2024
        assert info['source'] == 'exif'
        assert info['exif_datetime'] is not None
        assert info['file_mtime'] is not None

    def test_info_without_exif(self, temp_photo_no_exif):
        """Should get info using file mtime when no EXIF."""
        info = get_datetime_info(temp_photo_no_exif)

        assert info['datetime'] is not None
        assert info['source'] == 'file_mtime'
        assert info['exif_datetime'] is None
        assert info['file_mtime'] is not None


class TestIsFromYear:
    """Tests for is_from_year function."""

    def test_is_from_correct_year(self, temp_photo_with_exif):
        """Should return True for matching year."""
        assert is_from_year(temp_photo_with_exif, 2024) is True

    def test_is_from_wrong_year(self, temp_photo_with_exif):
        """Should return False for non-matching year."""
        assert is_from_year(temp_photo_with_exif, 2023) is False


class TestFilterByYear:
    """Tests for filter_by_year function."""

    def test_filter_photos_by_year(self, temp_photo_with_exif, temp_photo_no_exif):
        """Should filter photos by year."""
        photos = [temp_photo_with_exif, temp_photo_no_exif]

        # Filter for 2024
        filtered = filter_by_year(photos, 2024)

        # At least the EXIF photo should match
        assert temp_photo_with_exif in filtered


class TestGroupByMonth:
    """Tests for group_by_month function."""

    def test_group_photos_by_month(self, temp_photo_with_exif):
        """Should group photos by month."""
        photos = [temp_photo_with_exif]

        grouped = group_by_month(photos)

        assert 6 in grouped  # June
        assert temp_photo_with_exif in grouped[6]

    def test_empty_list(self):
        """Should handle empty list."""
        grouped = group_by_month([])
        assert grouped == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
