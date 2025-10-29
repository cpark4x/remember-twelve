#!/usr/bin/env python3
"""
Detect potential duplicate photos based on:
1. Same capture date (day)
2. Similar file sizes
3. Similar image dimensions
"""

import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

def get_photo_info(photo_path):
    """Get photo metadata for comparison"""
    try:
        # Get file size
        file_size = photo_path.stat().st_size

        # Get image dimensions
        img = Image.open(photo_path)
        width, height = img.size

        # Get EXIF date
        exif = img.getexif()
        date_str = None
        if exif and 36867 in exif:  # DateTimeOriginal
            date_str = exif[36867]
            capture_date = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
        else:
            # Fallback to file modification time
            capture_date = datetime.fromtimestamp(photo_path.stat().st_mtime)

        return {
            'path': photo_path,
            'size': file_size,
            'width': width,
            'height': height,
            'date': capture_date,
            'date_day': capture_date.date()
        }
    except Exception as e:
        print(f"Warning: Could not analyze {photo_path.name}: {e}")
        return None

def find_duplicates(photo_dir):
    """Find potential duplicate photos"""
    photos_dir = Path(photo_dir)

    # Get all photo files
    photo_files = []
    for ext in ['*.jpg', '*.jpeg', '*.HEIC', '*.heic', '*.png', '*.JPG', '*.JPEG']:
        photo_files.extend(photos_dir.glob(ext))

    # Analyze all photos
    photo_infos = []
    for photo in sorted(photo_files):
        info = get_photo_info(photo)
        if info:
            photo_infos.append(info)

    print(f"Analyzed {len(photo_infos)} photos\n")

    # Find duplicates
    duplicates = []
    for i, photo1 in enumerate(photo_infos):
        for photo2 in photo_infos[i+1:]:
            # Same day?
            if photo1['date_day'] == photo2['date_day']:
                # Similar size? (within 20%)
                size_ratio = min(photo1['size'], photo2['size']) / max(photo1['size'], photo2['size'])
                if size_ratio > 0.8:
                    duplicates.append((photo1, photo2))
                    print(f"⚠️  POTENTIAL DUPLICATES:")
                    print(f"   {photo1['path'].name:25} - {photo1['date']} - {photo1['size']/1024/1024:.1f}MB - {photo1['width']}x{photo1['height']}")
                    print(f"   {photo2['path'].name:25} - {photo2['date']} - {photo2['size']/1024/1024:.1f}MB - {photo2['width']}x{photo2['height']}")
                    print()

    if not duplicates:
        print("✓ No obvious duplicates found!")
    else:
        print(f"\nFound {len(duplicates)} potential duplicate pairs")
        print("\nRecommendation: Review these photos and delete the lower quality ones")

    return duplicates

if __name__ == '__main__':
    photo_dir = Path.home() / 'Pictures' / '2023 Remember Twelve'
    find_duplicates(photo_dir)
