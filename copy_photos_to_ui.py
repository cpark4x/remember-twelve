#!/usr/bin/env python3
"""
Copy curated photos to ui/photos directory for web viewing.
"""

import json
import shutil
from pathlib import Path

def copy_photos_to_ui(json_file: str = "twelve_2023_balanced.json"):
    """Copy curated photos to ui/photos directory."""

    # Load the curation data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create ui/photos directory
    photos_dir = Path("ui/photos")
    photos_dir.mkdir(parents=True, exist_ok=True)

    print(f"📸 Copying {len(data['photos'])} photos to ui/photos/")
    print("=" * 60)

    # Copy each photo with index as filename
    for i, photo in enumerate(data['photos']):
        src = Path(photo['photo_path'])

        if not src.exists():
            print(f"❌ {i}: Photo not found: {src.name}")
            continue

        # Use simple numbered names: 0.jpeg, 1.jpeg, etc.
        ext = src.suffix.lower()
        # Convert HEIC to jpeg extension (browser compatibility)
        if ext in ['.heic', '.heif']:
            ext = '.jpg'  # We'll keep original format, browser may handle it

        dest = photos_dir / f"{i}{src.suffix}"

        try:
            shutil.copy2(src, dest)
            print(f"✅ {i}: {src.name} → {dest.name}")
        except Exception as e:
            print(f"❌ {i}: Error copying: {e}")

    print("=" * 60)
    print(f"✅ Photos copied to: {photos_dir.absolute()}")
    print()

if __name__ == '__main__':
    copy_photos_to_ui()
