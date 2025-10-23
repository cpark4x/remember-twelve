#!/usr/bin/env python3
"""
Demo: Google Photos Integration - Phase 1 (Authentication & Basic Fetching)

This demo demonstrates:
1. OAuth 2.0 authentication with Google Photos
2. Listing photos from a specific year
3. Downloading photos to temporary cache
4. Extracting photo metadata
5. Getting Google Photos URLs

Prerequisites:
1. Google Cloud project with Photos Library API enabled
2. OAuth credentials downloaded as google_photos_credentials.json
3. Dependencies installed: pip install -r requirements.txt

Usage:
    python demo_google_photos_phase1.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from photo_sources import PhotoSourceFactory


def main():
    """Run Phase 1 authentication and basic fetching demo."""
    print("=" * 70)
    print("Google Photos Integration - Phase 1 Demo")
    print("Authentication & Basic Photo Fetching")
    print("=" * 70)
    print()

    # Check for credentials
    creds_path = Path('google_photos_credentials.json')
    if not creds_path.exists():
        print("ERROR: google_photos_credentials.json not found!")
        print()
        print("To get credentials:")
        print("1. Go to https://console.cloud.google.com")
        print("2. Create a project and enable Google Photos Library API")
        print("3. Create OAuth 2.0 credentials (Desktop app)")
        print("4. Download as google_photos_credentials.json")
        print()
        return

    # Create Google Photos source
    print("Step 1: Creating Google Photos source...")
    config = {
        'source_type': 'google_photos',
        'credentials_path': str(creds_path)
    }
    source = PhotoSourceFactory.create(config)
    print("✓ Source created")
    print()

    # Authenticate
    print("Step 2: Authenticating with Google Photos...")
    print("(Browser will open for OAuth consent)")
    print()

    try:
        user_email = source.authenticate()
        print(f"✓ Authenticated as: {user_email}")
        print()
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return

    # Fetch photos from 2023
    print("Step 3: Fetching photos from 2023...")
    print()

    try:
        photo_count = 0
        max_photos = 10  # Limit for demo

        for photo_path in source.scan(year=2023):
            photo_count += 1

            # Get metadata
            metadata = source.get_metadata(photo_path)
            url = source.get_original_url(photo_path)

            # Display info
            print(f"Photo #{photo_count}:")
            print(f"  Path: {Path(photo_path).name}")
            print(f"  Date: {metadata.get('timestamp', 'Unknown')}")
            print(f"  Size: {metadata.get('width')}x{metadata.get('height')}")
            print(f"  Format: {metadata.get('format')}")
            print(f"  File Size: {metadata.get('file_size') // 1024}KB")
            if url:
                print(f"  Google Photos: {url}")
            print()

            if photo_count >= max_photos:
                print(f"(Stopping at {max_photos} photos for demo)")
                break

        if photo_count == 0:
            print("No photos found in 2023")
        else:
            print(f"✓ Successfully fetched and downloaded {photo_count} photos")

    except Exception as e:
        print(f"✗ Error fetching photos: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print()
        print("Step 4: Cleaning up temporary cache...")
        source.cleanup()
        print("✓ Cache cleaned")

    print()
    print("=" * 70)
    print("Phase 1 Demo Complete!")
    print()
    print("Next Steps:")
    print("- Phase 2: Implement TempPhotoCache for better caching")
    print("- Phase 3: Integrate with TwelveCurator for full curation")
    print("=" * 70)


if __name__ == '__main__':
    main()
