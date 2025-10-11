"""
Photo Library Scanner

Discovers photos recursively in directory structures, with support for
common photo formats, file validation, and ignore patterns.

Key Features:
- Recursive directory scanning
- Support for common formats (.jpg, .jpeg, .png, .heic, .heif)
- File validation (size, format, readability)
- Ignore patterns (hidden files, system folders)
- Generator pattern for memory efficiency
- Scan statistics

Examples:
    >>> from photo_quality_analyzer import LibraryScanner
    >>> scanner = LibraryScanner()
    >>>
    >>> # Scan directory
    >>> for photo_path in scanner.scan('/path/to/photos'):
    ...     print(f"Found: {photo_path}")
    >>>
    >>> # Get statistics
    >>> stats = scanner.get_stats()
    >>> print(f"Found {stats['total_found']} photos")
"""

import logging
from pathlib import Path
from typing import Iterator, List, Set, Optional, Dict, Any
from dataclasses import dataclass, field

from .config import QualityAnalyzerConfig


logger = logging.getLogger(__name__)


@dataclass
class ScanStatistics:
    """
    Statistics from library scan operation.

    Attributes:
        total_found: Total number of photos found
        skipped: Number of files skipped (wrong format, too small, etc.)
        errors: Number of errors encountered
        directories_scanned: Number of directories scanned
        skipped_reasons: Dictionary mapping reason to count
    """
    total_found: int = 0
    skipped: int = 0
    errors: int = 0
    directories_scanned: int = 0
    skipped_reasons: Dict[str, int] = field(default_factory=dict)

    def add_skipped(self, reason: str) -> None:
        """Add a skipped file with reason."""
        self.skipped += 1
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_found': self.total_found,
            'skipped': self.skipped,
            'errors': self.errors,
            'directories_scanned': self.directories_scanned,
            'skipped_reasons': self.skipped_reasons
        }


class LibraryScanner:
    """
    Scanner for discovering photos in directory structures.

    Recursively scans directories for photo files, with support for
    filtering, validation, and ignore patterns. Uses generator pattern
    for memory-efficient operation on large photo libraries.

    Examples:
        >>> scanner = LibraryScanner()
        >>> photos = list(scanner.scan('/Users/john/Photos'))
        >>> print(f"Found {len(photos)} photos")
        >>>
        >>> stats = scanner.get_stats()
        >>> print(f"Scanned {stats.directories_scanned} directories")
    """

    # Supported photo file extensions
    DEFAULT_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}

    # Patterns to ignore
    DEFAULT_IGNORE_PATTERNS = {
        # Hidden files/folders
        '.*',
        # System folders
        '__MACOSX',
        'node_modules',
        '.git',
        '.DS_Store',
        'Thumbs.db',
        # Common non-photo files
        '.txt',
        '.json',
        '.xml'
    }

    def __init__(
        self,
        config: Optional[QualityAnalyzerConfig] = None,
        extensions: Optional[Set[str]] = None,
        ignore_patterns: Optional[Set[str]] = None,
        min_file_size: int = 1024,  # 1KB minimum
        max_file_size: Optional[int] = None
    ):
        """
        Initialize library scanner.

        Args:
            config: Quality analyzer configuration
            extensions: Set of file extensions to scan (defaults to common photo formats)
            ignore_patterns: Patterns to ignore (defaults to system files/folders)
            min_file_size: Minimum file size in bytes (default 1KB)
            max_file_size: Maximum file size in bytes (default unlimited)
        """
        self.config = config or QualityAnalyzerConfig()
        self.extensions = extensions or self.DEFAULT_EXTENSIONS
        self.ignore_patterns = ignore_patterns or self.DEFAULT_IGNORE_PATTERNS
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size

        self._stats = ScanStatistics()

    def scan(
        self,
        root_path: str,
        recursive: bool = True
    ) -> Iterator[str]:
        """
        Scan directory for photos.

        Yields photo file paths as they are discovered. Uses generator
        pattern for memory efficiency.

        Args:
            root_path: Root directory to scan
            recursive: Whether to scan subdirectories (default True)

        Yields:
            str: Absolute path to photo file

        Examples:
            >>> scanner = LibraryScanner()
            >>> for photo in scanner.scan('/Users/john/Photos'):
            ...     print(f"Found: {photo}")
            >>>
            >>> # Non-recursive scan
            >>> for photo in scanner.scan('/Users/john/Photos', recursive=False):
            ...     print(f"Found: {photo}")
        """
        root = Path(root_path)

        if not root.exists():
            logger.error(f"Scan path does not exist: {root_path}")
            self._stats.errors += 1
            return

        if not root.is_dir():
            logger.error(f"Scan path is not a directory: {root_path}")
            self._stats.errors += 1
            return

        logger.info(f"Starting scan of: {root_path} (recursive={recursive})")

        # Reset statistics
        self._stats = ScanStatistics()

        # Scan directory
        if recursive:
            yield from self._scan_recursive(root)
        else:
            yield from self._scan_directory(root)

        logger.info(
            f"Scan complete: {self._stats.total_found} photos found, "
            f"{self._stats.skipped} skipped, {self._stats.errors} errors"
        )

    def _scan_recursive(self, root: Path) -> Iterator[str]:
        """
        Recursively scan directory tree.

        Args:
            root: Root directory path

        Yields:
            str: Absolute path to photo file
        """
        try:
            for item in root.iterdir():
                # Skip ignored patterns
                if self._should_ignore(item):
                    continue

                if item.is_dir():
                    # Recursively scan subdirectory
                    self._stats.directories_scanned += 1
                    yield from self._scan_recursive(item)

                elif item.is_file():
                    # Check if it's a valid photo
                    if self._is_valid_photo(item):
                        self._stats.total_found += 1
                        yield str(item.absolute())

        except PermissionError:
            logger.warning(f"Permission denied: {root}")
            self._stats.errors += 1
        except Exception as e:
            logger.error(f"Error scanning {root}: {e}")
            self._stats.errors += 1

    def _scan_directory(self, directory: Path) -> Iterator[str]:
        """
        Scan single directory (non-recursive).

        Args:
            directory: Directory path

        Yields:
            str: Absolute path to photo file
        """
        try:
            self._stats.directories_scanned += 1

            for item in directory.iterdir():
                # Skip ignored patterns
                if self._should_ignore(item):
                    continue

                if item.is_file():
                    # Check if it's a valid photo
                    if self._is_valid_photo(item):
                        self._stats.total_found += 1
                        yield str(item.absolute())

        except PermissionError:
            logger.warning(f"Permission denied: {directory}")
            self._stats.errors += 1
        except Exception as e:
            logger.error(f"Error scanning {directory}: {e}")
            self._stats.errors += 1

    def _should_ignore(self, path: Path) -> bool:
        """
        Check if path should be ignored based on ignore patterns.

        Args:
            path: File or directory path

        Returns:
            True if should be ignored, False otherwise
        """
        name = path.name

        # Check exact matches
        if name in self.ignore_patterns:
            return True

        # Check pattern matches (e.g., '.*' for hidden files)
        for pattern in self.ignore_patterns:
            if pattern.startswith('.') and pattern.endswith('*'):
                # Hidden file pattern
                if name.startswith('.'):
                    return True
            elif pattern.endswith('*'):
                # Prefix pattern
                prefix = pattern[:-1]
                if name.startswith(prefix):
                    return True

        return False

    def _is_valid_photo(self, path: Path) -> bool:
        """
        Check if file is a valid photo.

        Validates extension, file size, and readability.

        Args:
            path: File path

        Returns:
            True if valid photo, False otherwise
        """
        # Check extension
        extension = path.suffix.lower()
        if extension not in self.extensions:
            self._stats.add_skipped(f"wrong_extension_{extension}")
            return False

        # Check file size
        try:
            file_size = path.stat().st_size

            if file_size < self.min_file_size:
                self._stats.add_skipped("too_small")
                logger.debug(f"File too small: {path} ({file_size} bytes)")
                return False

            if self.max_file_size and file_size > self.max_file_size:
                self._stats.add_skipped("too_large")
                logger.debug(f"File too large: {path} ({file_size} bytes)")
                return False

        except OSError as e:
            logger.warning(f"Error checking file size: {path} - {e}")
            self._stats.add_skipped("size_check_error")
            return False

        # Check readability
        try:
            with open(path, 'rb') as f:
                # Try to read first few bytes
                header = f.read(16)
                if len(header) == 0:
                    self._stats.add_skipped("empty_file")
                    return False

        except (IOError, OSError) as e:
            logger.warning(f"Cannot read file: {path} - {e}")
            self._stats.add_skipped("read_error")
            return False

        return True

    def get_stats(self) -> ScanStatistics:
        """
        Get scan statistics.

        Returns:
            ScanStatistics object with scan results

        Examples:
            >>> scanner = LibraryScanner()
            >>> photos = list(scanner.scan('/path/to/photos'))
            >>> stats = scanner.get_stats()
            >>> print(f"Found {stats.total_found} photos")
            >>> print(f"Skipped {stats.skipped} files")
        """
        return self._stats

    def scan_and_collect(
        self,
        root_path: str,
        recursive: bool = True,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Scan directory and collect all photo paths into a list.

        Convenience method that collects all paths instead of yielding.
        Useful for smaller libraries or when you need the full list.

        Args:
            root_path: Root directory to scan
            recursive: Whether to scan subdirectories (default True)
            limit: Maximum number of photos to collect (default unlimited)

        Returns:
            List of photo file paths

        Examples:
            >>> scanner = LibraryScanner()
            >>> photos = scanner.scan_and_collect('/Users/john/Photos')
            >>> print(f"Found {len(photos)} photos")
            >>>
            >>> # Limit results
            >>> first_100 = scanner.scan_and_collect('/Users/john/Photos', limit=100)
        """
        photos = []

        for photo_path in self.scan(root_path, recursive=recursive):
            photos.append(photo_path)

            if limit and len(photos) >= limit:
                logger.info(f"Reached limit of {limit} photos")
                break

        return photos

    def scan_multiple(
        self,
        root_paths: List[str],
        recursive: bool = True
    ) -> Iterator[str]:
        """
        Scan multiple directories.

        Args:
            root_paths: List of root directories to scan
            recursive: Whether to scan subdirectories (default True)

        Yields:
            str: Absolute path to photo file

        Examples:
            >>> scanner = LibraryScanner()
            >>> paths = ['/Users/john/Photos', '/Users/john/Pictures']
            >>> for photo in scanner.scan_multiple(paths):
            ...     print(f"Found: {photo}")
        """
        for root_path in root_paths:
            logger.info(f"Scanning: {root_path}")
            yield from self.scan(root_path, recursive=recursive)

    def estimate_count(self, root_path: str, sample_depth: int = 2) -> int:
        """
        Estimate total number of photos without full scan.

        Samples subdirectories to estimate total count. Useful for
        showing progress estimates before full scan.

        Args:
            root_path: Root directory to estimate
            sample_depth: How many directory levels to sample (default 2)

        Returns:
            Estimated photo count

        Examples:
            >>> scanner = LibraryScanner()
            >>> estimated = scanner.estimate_count('/Users/john/Photos')
            >>> print(f"Estimated {estimated} photos")
        """
        root = Path(root_path)

        if not root.exists() or not root.is_dir():
            return 0

        try:
            # Count photos at current level
            photos_here = sum(
                1 for item in root.iterdir()
                if item.is_file() and self._is_valid_photo(item)
            )

            # If we've reached sample depth, just return count here
            if sample_depth <= 0:
                return photos_here

            # Sample subdirectories
            subdirs = [item for item in root.iterdir() if item.is_dir() and not self._should_ignore(item)]

            if not subdirs:
                return photos_here

            # Sample up to 5 subdirectories
            sample_size = min(5, len(subdirs))
            sample_dirs = subdirs[:sample_size]

            # Recursively estimate subdirectories
            subdir_estimates = [
                self.estimate_count(str(d), sample_depth - 1)
                for d in sample_dirs
            ]

            # Average estimate per subdirectory
            avg_per_subdir = sum(subdir_estimates) / sample_size if sample_size > 0 else 0

            # Extrapolate to all subdirectories
            total_estimate = photos_here + (avg_per_subdir * len(subdirs))

            return int(total_estimate)

        except Exception as e:
            logger.warning(f"Error estimating count: {e}")
            return 0

    def validate_paths(self, photo_paths: List[str]) -> List[str]:
        """
        Validate a list of photo paths.

        Filters out invalid paths (non-existent, wrong format, etc.)

        Args:
            photo_paths: List of photo file paths

        Returns:
            List of valid photo paths

        Examples:
            >>> scanner = LibraryScanner()
            >>> valid = scanner.validate_paths([
            ...     '/path/to/photo1.jpg',
            ...     '/path/to/nonexistent.jpg',
            ...     '/path/to/photo2.png'
            ... ])
        """
        valid_paths = []

        for path in photo_paths:
            path_obj = Path(path)

            if not path_obj.exists():
                self._stats.add_skipped("not_found")
                continue

            if not path_obj.is_file():
                self._stats.add_skipped("not_a_file")
                continue

            if self._is_valid_photo(path_obj):
                valid_paths.append(path)

        return valid_paths
