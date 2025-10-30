"""
Apple Photos Export Helper

Provides automated export from Apple Photos app via AppleScript (macOS only).
Falls back to manual instructions on other platforms.

This is a lightweight helper that automates the export workflow without
requiring native PhotoKit integration. Works with existing LocalPhotoSource.
"""

import subprocess
import platform
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime


class ApplePhotosNotAvailableError(Exception):
    """Raised when Apple Photos is not available on this system."""
    pass


class ApplePhotosHelper:
    """
    Helper for exporting photos from Apple Photos app.

    On macOS: Automates export via AppleScript
    On other platforms: Provides manual instructions
    """

    @staticmethod
    def is_available() -> bool:
        """Check if Apple Photos automation is available."""
        if platform.system() != 'Darwin':  # Not macOS
            return False

        # Check if Photos.app exists
        photos_app = Path('/Applications/Photos.app')
        return photos_app.exists()

    @staticmethod
    def export_photos(
        year: int,
        output_dir: Optional[Path] = None,
        album_name: Optional[str] = None
    ) -> Path:
        """
        Export photos from Apple Photos app.

        Args:
            year: Year to export (e.g., 2023)
            output_dir: Where to export (uses temp dir if None)
            album_name: Optional album name to export from

        Returns:
            Path to exported photos directory

        Raises:
            ApplePhotosNotAvailableError: If Photos.app not available
            subprocess.CalledProcessError: If AppleScript fails
        """
        if not ApplePhotosHelper.is_available():
            raise ApplePhotosNotAvailableError(
                "Apple Photos is not available on this system. "
                "Use manual export method (see docs/APPLE_PHOTOS_GUIDE.md)"
            )

        # Create output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix=f'remember_twelve_photos_{year}_'))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📤 Exporting photos from Apple Photos...")
        print(f"   Year: {year}")
        if album_name:
            print(f"   Album: {album_name}")
        print(f"   Destination: {output_dir}")
        print()

        # Build AppleScript
        script = ApplePhotosHelper._build_applescript(year, output_dir, album_name)

        try:
            # Execute AppleScript
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for large libraries
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip()
                print(f"❌ Export failed: {error_msg}")
                raise subprocess.CalledProcessError(
                    result.returncode,
                    'osascript',
                    output=result.stdout,
                    stderr=result.stderr
                )

            print("✅ Export complete!")
            print(f"   Photos exported to: {output_dir}")
            print()

            return output_dir

        except subprocess.TimeoutExpired:
            print("❌ Export timed out (>10 minutes)")
            print("   Try exporting a smaller date range or specific album")
            raise

    @staticmethod
    def _build_applescript(
        year: int,
        output_dir: Path,
        album_name: Optional[str] = None
    ) -> str:
        """Build AppleScript for exporting photos."""

        # Date range for year
        start_date = f"1/1/{year}"
        end_date = f"1/1/{year + 1}"

        if album_name:
            # Export from specific album
            script = f'''
tell application "Photos"
    set exportFolder to POSIX file "{output_dir}"

    try
        set theAlbum to album "{album_name}"
        set thePhotos to every media item of theAlbum

        if (count of thePhotos) is 0 then
            error "No photos found in album '{album_name}'"
        end if

        export thePhotos to exportFolder
    on error errMsg
        error "Failed to export from album '{album_name}': " & errMsg
    end try
end tell
'''
        else:
            # Export all photos from year
            script = f'''
tell application "Photos"
    set exportFolder to POSIX file "{output_dir}"
    set startDate to date "{start_date}"
    set endDate to date "{end_date}"

    set thePhotos to every media item whose date is greater than or equal to startDate and date is less than endDate

    if (count of thePhotos) is 0 then
        error "No photos found for year {year}"
    end if

    export thePhotos to exportFolder
end tell
'''

        return script

    @staticmethod
    def get_manual_instructions(year: int, album_name: Optional[str] = None) -> str:
        """Get manual export instructions for users."""

        if album_name:
            instructions = f"""
📋 Manual Export Instructions for Apple Photos

Album: {album_name}

1. Open Photos app
2. Click "{album_name}" album in sidebar
3. Select all photos: Cmd + A (or Edit → Select All)
4. Export photos:
   • File → Export → Export Unmodified Originals
   • Choose destination folder
   • Ensure "Include Metadata" is checked
   • Click "Export"
5. Once exported, run:

   ./sync_photos.sh <exported-folder-path> {year}

📖 Full guide: docs/APPLE_PHOTOS_GUIDE.md
"""
        else:
            instructions = f"""
📋 Manual Export Instructions for Apple Photos

Year: {year}

1. Open Photos app
2. Filter by year:
   • Click search icon (🔍)
   • Type: date:{year}
   • Press Enter
3. Select all photos: Cmd + A (or Edit → Select All)
4. Export photos:
   • File → Export → Export Unmodified Originals
   • Choose destination folder (e.g., ~/Desktop/Photos{year})
   • Ensure "Include Metadata" is checked
   • Click "Export"
5. Once exported, run:

   ./sync_photos.sh ~/Desktop/Photos{year} {year}

📖 Full guide: docs/APPLE_PHOTOS_GUIDE.md
"""

        return instructions.strip()

    @staticmethod
    def show_manual_instructions(year: int, album_name: Optional[str] = None):
        """Print manual export instructions."""
        print(ApplePhotosHelper.get_manual_instructions(year, album_name))
        print()


# Convenience function for CLI usage
def export_from_apple_photos(
    year: int,
    output_dir: Optional[Path] = None,
    album_name: Optional[str] = None,
    manual: bool = False
) -> Tuple[Path, bool]:
    """
    Export photos from Apple Photos (automated or manual).

    Args:
        year: Year to export
        output_dir: Where to export (temp dir if None)
        album_name: Optional album to export from
        manual: Force manual instructions instead of automation

    Returns:
        Tuple of (export_path, was_automated)
        - If automated: (path, True)
        - If manual: (None, False) - user must export manually
    """
    if manual or not ApplePhotosHelper.is_available():
        # Show manual instructions
        print("🍎 Apple Photos Export")
        print()

        if not ApplePhotosHelper.is_available():
            if platform.system() != 'Darwin':
                print("⚠️  Not running on macOS")
            else:
                print("⚠️  Photos.app not found")
            print("   Using manual export method")
            print()

        ApplePhotosHelper.show_manual_instructions(year, album_name)
        return None, False

    # Automated export
    try:
        export_path = ApplePhotosHelper.export_photos(year, output_dir, album_name)
        return export_path, True

    except (ApplePhotosNotAvailableError, subprocess.CalledProcessError) as e:
        print(f"⚠️  Automated export failed: {e}")
        print()
        print("Falling back to manual instructions:")
        print()
        ApplePhotosHelper.show_manual_instructions(year, album_name)
        return None, False
