#!/usr/bin/env python3
"""
Quick Test - Validate Google Photos Integration

This is a quick validation that doesn't require full curation.
Tests authentication and basic photo fetching only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from photo_sources import PhotoSourceFactory

def main():
    print("=" * 60)
    print("Quick Test: Google Photos Integration")
    print("=" * 60)
    print()

    # Check credentials
    creds_path = Path('google_photos_credentials.json')
    if not creds_path.exists():
        print("ERROR: google_photos_credentials.json not found!")
        return 1

    print("✓ Credentials file found")
    print()

    # Create source
    print("Creating Google Photos source...")
    try:
        source = PhotoSourceFactory.create_google_photos(str(creds_path))
        print("✓ Source created")
        print()
    except Exception as e:
        print(f"✗ Failed: {e}")
        return 1

    # Authenticate
    print("Authenticating (browser may open)...")
    try:
        user_email = source.authenticate()
        print(f"✓ Authenticated as: {user_email}")
        print()
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return 1

    # Fetch a few photos from 2023
    print("Fetching first 5 photos from 2023...")
    try:
        count = 0
        for photo_path in source.scan(year=2023):
            count += 1
            metadata = source.get_metadata(photo_path)
            url = source.get_original_url(photo_path)

            print(f"  {count}. {Path(photo_path).name}")
            print(f"     Date: {metadata.get('timestamp')}")
            print(f"     Size: {metadata.get('width')}x{metadata.get('height')}")
            if url:
                print(f"     URL: {url[:60]}...")
            print()

            if count >= 5:
                break

        source.cleanup()
        print(f"✓ Successfully fetched {count} photos")
        print()

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 60)
    print("SUCCESS! All components working.")
    print("=" * 60)
    print()
    print("Ready for full curation:")
    print("  python3 curate_from_google_photos.py --year 2023")
    print()

    return 0

if __name__ == '__main__':
    sys.exit(main())
