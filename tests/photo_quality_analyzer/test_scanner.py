"""
Tests for LibraryScanner

Covers directory scanning, file validation, ignore patterns,
statistics tracking, and error handling.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from src.photo_quality_analyzer.scanner import LibraryScanner, ScanStatistics


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def photo_library(temp_dir):
    """
    Create a test photo library structure:

    temp_dir/
    ├── photo1.jpg
    ├── photo2.png
    ├── document.txt (non-photo)
    ├── subfolder1/
    │   ├── photo3.jpg
    │   └── photo4.jpeg
    ├── subfolder2/
    │   └── photo5.heic
    └── .hidden/
        └── photo6.jpg (hidden)
    """
    root = Path(temp_dir)

    # Create photos at root
    for i in [1, 2]:
        ext = '.jpg' if i == 1 else '.png'
        photo_path = root / f"photo{i}{ext}"
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(image).save(photo_path)

    # Create non-photo file
    (root / "document.txt").write_text("This is not a photo")

    # Create subfolder1 with photos
    subfolder1 = root / "subfolder1"
    subfolder1.mkdir()
    for i in [3, 4]:
        ext = '.jpg' if i == 3 else '.jpeg'
        photo_path = subfolder1 / f"photo{i}{ext}"
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(image).save(photo_path)

    # Create subfolder2 with photo
    subfolder2 = root / "subfolder2"
    subfolder2.mkdir()
    # Note: .heic files might not be supported by PIL, so we'll create a dummy file
    # with the extension for testing purposes
    heic_path = subfolder2 / "photo5.heic"
    heic_path.write_bytes(b"HEIC header simulation" * 100)  # Dummy content

    # Create hidden folder with photo (should be ignored by default)
    hidden = root / ".hidden"
    hidden.mkdir()
    photo_path = hidden / "photo6.jpg"
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(image).save(photo_path)

    return root


@pytest.fixture
def scanner():
    """Create scanner instance."""
    return LibraryScanner()


class TestScanStatistics:
    """Test ScanStatistics dataclass."""

    def test_statistics_creation(self):
        """Test creating scan statistics."""
        stats = ScanStatistics()

        assert stats.total_found == 0
        assert stats.skipped == 0
        assert stats.errors == 0
        assert stats.directories_scanned == 0
        assert len(stats.skipped_reasons) == 0

    def test_add_skipped(self):
        """Test adding skipped files."""
        stats = ScanStatistics()

        stats.add_skipped("wrong_extension")
        stats.add_skipped("too_small")
        stats.add_skipped("wrong_extension")

        assert stats.skipped == 3
        assert stats.skipped_reasons["wrong_extension"] == 2
        assert stats.skipped_reasons["too_small"] == 1

    def test_to_dict(self):
        """Test converting statistics to dictionary."""
        stats = ScanStatistics()
        stats.total_found = 10
        # Manually set skipped to 5, then add_skipped will increment it
        stats.skipped = 5
        stats.add_skipped("too_small")

        result = stats.to_dict()

        assert result['total_found'] == 10
        assert result['skipped'] == 6  # 5 + 1 from add_skipped
        assert 'skipped_reasons' in result


class TestLibraryScannerInitialization:
    """Test scanner initialization."""

    def test_default_initialization(self):
        """Test scanner with default settings."""
        scanner = LibraryScanner()

        assert scanner.extensions == LibraryScanner.DEFAULT_EXTENSIONS
        assert scanner.ignore_patterns == LibraryScanner.DEFAULT_IGNORE_PATTERNS
        assert scanner.min_file_size == 1024
        assert scanner.max_file_size is None

    def test_custom_extensions(self):
        """Test scanner with custom extensions."""
        scanner = LibraryScanner(extensions={'.jpg', '.png'})

        assert scanner.extensions == {'.jpg', '.png'}
        assert '.heic' not in scanner.extensions

    def test_custom_ignore_patterns(self):
        """Test scanner with custom ignore patterns."""
        scanner = LibraryScanner(ignore_patterns={'temp*', '*.bak'})

        assert scanner.ignore_patterns == {'temp*', '*.bak'}

    def test_custom_file_size_limits(self):
        """Test scanner with custom file size limits."""
        scanner = LibraryScanner(
            min_file_size=2048,
            max_file_size=10485760
        )

        assert scanner.min_file_size == 2048
        assert scanner.max_file_size == 10485760


class TestLibraryScanning:
    """Test library scanning functionality."""

    def test_scan_empty_directory(self, scanner, temp_dir):
        """Test scanning an empty directory."""
        photos = list(scanner.scan(temp_dir))

        assert len(photos) == 0

        stats = scanner.get_stats()
        assert stats.total_found == 0

    def test_scan_nonexistent_directory(self, scanner):
        """Test scanning a non-existent directory."""
        photos = list(scanner.scan("/nonexistent/path"))

        assert len(photos) == 0

        stats = scanner.get_stats()
        assert stats.errors > 0

    def test_scan_file_not_directory(self, scanner, temp_dir):
        """Test scanning a file instead of directory."""
        file_path = Path(temp_dir) / "file.txt"
        file_path.write_text("test")

        photos = list(scanner.scan(str(file_path)))

        assert len(photos) == 0

        stats = scanner.get_stats()
        assert stats.errors > 0

    def test_scan_recursive(self, scanner, photo_library):
        """Test recursive directory scanning."""
        photos = list(scanner.scan(str(photo_library), recursive=True))

        # Should find: photo1.jpg, photo2.png, photo3.jpg, photo4.jpeg
        # Not photo5.heic (if not supported by image library)
        # Not photo6.jpg (hidden folder)
        assert len(photos) >= 4

        # Verify paths are absolute
        for photo in photos:
            assert Path(photo).is_absolute()

        stats = scanner.get_stats()
        assert stats.total_found >= 4
        assert stats.directories_scanned > 1

    def test_scan_non_recursive(self, scanner, photo_library):
        """Test non-recursive directory scanning."""
        photos = list(scanner.scan(str(photo_library), recursive=False))

        # Should only find photo1.jpg and photo2.png at root level
        assert len(photos) == 2

        stats = scanner.get_stats()
        assert stats.total_found == 2
        assert stats.directories_scanned == 1

    def test_scan_ignores_hidden_files(self, scanner, photo_library):
        """Test that hidden files/folders are ignored."""
        photos = list(scanner.scan(str(photo_library), recursive=True))

        # Should not include .hidden/photo6.jpg
        hidden_photo = str(photo_library / ".hidden" / "photo6.jpg")
        assert hidden_photo not in photos

    def test_scan_skips_non_photos(self, scanner, photo_library):
        """Test that non-photo files are skipped."""
        photos = list(scanner.scan(str(photo_library), recursive=True))

        # Should not include document.txt
        txt_file = str(photo_library / "document.txt")
        assert txt_file not in photos

        stats = scanner.get_stats()
        assert stats.skipped > 0


class TestFileValidation:
    """Test file validation logic."""

    def test_validate_extension(self, scanner, temp_dir):
        """Test extension validation."""
        # Valid extension
        jpg_path = Path(temp_dir) / "photo.jpg"
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(image).save(jpg_path)

        photos = list(scanner.scan(temp_dir))
        assert len(photos) == 1

        # Invalid extension
        txt_path = Path(temp_dir) / "document.txt"
        txt_path.write_text("not a photo")

        scanner._stats = ScanStatistics()  # Reset stats
        photos = list(scanner.scan(temp_dir))

        stats = scanner.get_stats()
        assert stats.skipped > 0

    def test_validate_min_file_size(self, temp_dir):
        """Test minimum file size validation."""
        scanner = LibraryScanner(min_file_size=5000)  # 5KB minimum

        # Create small file
        small_photo = Path(temp_dir) / "small.jpg"
        small_photo.write_bytes(b"JPEG" * 10)  # Very small

        photos = list(scanner.scan(temp_dir))

        assert len(photos) == 0

        stats = scanner.get_stats()
        assert stats.skipped > 0

    def test_validate_max_file_size(self, temp_dir):
        """Test maximum file size validation."""
        scanner = LibraryScanner(max_file_size=1000)  # 1KB maximum

        # Create normal-sized photo (will be > 1KB)
        photo_path = Path(temp_dir) / "large.jpg"
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        Image.fromarray(image).save(photo_path)

        photos = list(scanner.scan(temp_dir))

        assert len(photos) == 0

        stats = scanner.get_stats()
        assert stats.skipped > 0

    def test_validate_empty_file(self, scanner, temp_dir):
        """Test that empty files are skipped."""
        empty_photo = Path(temp_dir) / "empty.jpg"
        empty_photo.touch()

        photos = list(scanner.scan(temp_dir))

        assert len(photos) == 0

        stats = scanner.get_stats()
        assert stats.skipped > 0


class TestIgnorePatterns:
    """Test ignore pattern functionality."""

    def test_ignore_hidden_files(self, temp_dir):
        """Test ignoring hidden files."""
        scanner = LibraryScanner()

        # Create hidden file
        hidden_photo = Path(temp_dir) / ".hidden_photo.jpg"
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(image).save(hidden_photo)

        photos = list(scanner.scan(temp_dir))

        assert len(photos) == 0

    def test_ignore_system_folders(self, temp_dir):
        """Test ignoring system folders."""
        scanner = LibraryScanner()

        # Create __MACOSX folder with photo
        macosx = Path(temp_dir) / "__MACOSX"
        macosx.mkdir()
        photo_path = macosx / "photo.jpg"
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(image).save(photo_path)

        photos = list(scanner.scan(temp_dir, recursive=True))

        assert len(photos) == 0

    def test_custom_ignore_patterns(self, temp_dir):
        """Test custom ignore patterns."""
        scanner = LibraryScanner(ignore_patterns={'temp*'})

        # Create temp folder
        temp_folder = Path(temp_dir) / "temp_photos"
        temp_folder.mkdir()
        photo_path = temp_folder / "photo.jpg"
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(image).save(photo_path)

        photos = list(scanner.scan(temp_dir, recursive=True))

        # Should be ignored
        assert len(photos) == 0


class TestScanAndCollect:
    """Test scan_and_collect convenience method."""

    def test_scan_and_collect(self, scanner, photo_library):
        """Test collecting all photos into a list."""
        photos = scanner.scan_and_collect(str(photo_library))

        assert isinstance(photos, list)
        assert len(photos) >= 4

    def test_scan_and_collect_with_limit(self, scanner, photo_library):
        """Test collecting with limit."""
        photos = scanner.scan_and_collect(str(photo_library), limit=2)

        assert len(photos) == 2

    def test_scan_and_collect_non_recursive(self, scanner, photo_library):
        """Test non-recursive collection."""
        photos = scanner.scan_and_collect(
            str(photo_library),
            recursive=False
        )

        assert len(photos) == 2  # Only root level photos


class TestScanMultiple:
    """Test scanning multiple directories."""

    def test_scan_multiple_directories(self, scanner, temp_dir):
        """Test scanning multiple directories."""
        # Create two separate directories with photos
        dir1 = Path(temp_dir) / "dir1"
        dir2 = Path(temp_dir) / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # Add photos to each
        for i, directory in enumerate([dir1, dir2]):
            photo_path = directory / f"photo_{i}.jpg"
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(image).save(photo_path)

        # Scan both
        photos = list(scanner.scan_multiple([str(dir1), str(dir2)]))

        assert len(photos) == 2


class TestEstimateCount:
    """Test photo count estimation."""

    def test_estimate_count(self, scanner, photo_library):
        """Test estimating photo count."""
        estimated = scanner.estimate_count(str(photo_library))

        # Should be reasonably close to actual count (4-5 photos)
        assert estimated >= 2  # At least some estimate

    def test_estimate_count_empty_directory(self, scanner, temp_dir):
        """Test estimating count for empty directory."""
        estimated = scanner.estimate_count(temp_dir)

        assert estimated == 0

    def test_estimate_count_nonexistent(self, scanner):
        """Test estimating count for non-existent directory."""
        estimated = scanner.estimate_count("/nonexistent/path")

        assert estimated == 0

    def test_estimate_count_depth(self, scanner, temp_dir):
        """Test estimation with different sample depths."""
        # Create nested structure
        root = Path(temp_dir)
        for i in range(3):
            subdir = root / f"level{i}"
            subdir.mkdir()
            photo_path = subdir / f"photo{i}.jpg"
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(image).save(photo_path)
            root = subdir

        estimate_deep = scanner.estimate_count(temp_dir, sample_depth=3)
        estimate_shallow = scanner.estimate_count(temp_dir, sample_depth=1)

        # Both should give some estimate
        assert estimate_deep >= 0
        assert estimate_shallow >= 0


class TestValidatePaths:
    """Test path validation."""

    def test_validate_paths_all_valid(self, scanner, photo_library):
        """Test validating all valid paths."""
        # Collect valid photos
        valid_photos = list(scanner.scan(str(photo_library)))

        # Reset stats and validate
        scanner._stats = ScanStatistics()
        validated = scanner.validate_paths(valid_photos)

        assert len(validated) == len(valid_photos)

    def test_validate_paths_mixed(self, scanner, photo_library):
        """Test validating mix of valid and invalid paths."""
        valid_photos = list(scanner.scan(str(photo_library)))

        mixed_paths = valid_photos + [
            "/nonexistent/photo.jpg",
            str(photo_library / "document.txt")
        ]

        scanner._stats = ScanStatistics()
        validated = scanner.validate_paths(mixed_paths)

        # Should only include valid photos
        assert len(validated) == len(valid_photos)

    def test_validate_paths_all_invalid(self, scanner):
        """Test validating all invalid paths."""
        invalid_paths = [
            "/nonexistent/photo1.jpg",
            "/nonexistent/photo2.jpg"
        ]

        validated = scanner.validate_paths(invalid_paths)

        assert len(validated) == 0


class TestScannerStatistics:
    """Test statistics tracking."""

    def test_statistics_tracking(self, scanner, photo_library):
        """Test that statistics are tracked correctly."""
        photos = list(scanner.scan(str(photo_library), recursive=True))

        stats = scanner.get_stats()

        assert stats.total_found == len(photos)
        assert stats.directories_scanned > 0
        assert stats.skipped >= 0  # May skip some files

    def test_statistics_reset_between_scans(self, scanner, photo_library):
        """Test that statistics reset between scans."""
        # First scan
        list(scanner.scan(str(photo_library)))
        stats1 = scanner.get_stats()

        # Second scan (stats should reset)
        list(scanner.scan(str(photo_library)))
        stats2 = scanner.get_stats()

        # Statistics should be same (not cumulative)
        assert stats1.total_found == stats2.total_found


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_permission_denied(self, scanner, temp_dir):
        """Test handling of permission denied errors."""
        # Create a directory with no read permissions
        restricted = Path(temp_dir) / "restricted"
        restricted.mkdir()

        # Add a photo
        photo_path = restricted / "photo.jpg"
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(image).save(photo_path)

        # Remove read permissions
        import os
        try:
            os.chmod(restricted, 0o000)

            # Scan should handle gracefully
            photos = list(scanner.scan(temp_dir, recursive=True))

            # May or may not find the photo depending on OS permissions
            # At minimum, shouldn't crash
            assert isinstance(photos, list)
        finally:
            # Restore permissions for cleanup
            os.chmod(restricted, 0o755)

    def test_symbolic_links(self, scanner, temp_dir):
        """Test handling of symbolic links."""
        # Create a photo
        real_dir = Path(temp_dir) / "real"
        real_dir.mkdir()
        photo_path = real_dir / "photo.jpg"
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(image).save(photo_path)

        # Create symlink to the directory
        link_dir = Path(temp_dir) / "link"
        try:
            link_dir.symlink_to(real_dir)

            # Scan should handle symlinks
            photos = list(scanner.scan(temp_dir, recursive=True))

            # Should find at least one photo (either original or through symlink)
            assert len(photos) >= 1
        except OSError:
            # Symlink creation might fail on some systems
            pytest.skip("Symlinks not supported on this system")

    def test_very_long_paths(self, scanner, temp_dir):
        """Test handling of very long file paths."""
        # Create nested directory structure
        long_path = Path(temp_dir)
        for i in range(10):
            long_path = long_path / f"very_long_directory_name_{i}"

        try:
            long_path.mkdir(parents=True)
            photo_path = long_path / "photo.jpg"
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(image).save(photo_path)

            # Scan should handle long paths
            photos = list(scanner.scan(temp_dir, recursive=True))

            assert len(photos) >= 1
        except OSError:
            # Path might be too long on some systems
            pytest.skip("Path too long for this system")
