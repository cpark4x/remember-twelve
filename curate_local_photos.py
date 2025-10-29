#!/usr/bin/env python3
"""
Curate photos from local Pictures directory with flexible month distribution.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
import pillow_heif

# Register HEIF opener with PIL
pillow_heif.register_heif_opener()

sys.path.insert(0, 'src')

from twelve_curator import TwelveCurator
from twelve_curator.data_classes import CurationConfig

def get_photo_date(photo_path):
    """Extract date from photo EXIF or file metadata."""
    try:
        img = Image.open(photo_path)
        exif = img.getexif()

        if exif:
            # Try DateTimeOriginal first (when photo was taken)
            datetime_original = exif.get(36867)  # DateTimeOriginal tag
            if datetime_original:
                return datetime.strptime(datetime_original, '%Y:%m:%d %H:%M:%S')

            # Fallback to DateTime
            datetime_tag = exif.get(306)  # DateTime tag
            if datetime_tag:
                return datetime.strptime(datetime_tag, '%Y:%m:%d %H:%M:%S')
    except Exception as e:
        print(f"  Warning: Could not read EXIF from {photo_path.name}: {e}")

    # Fallback to file modification time
    return datetime.fromtimestamp(photo_path.stat().st_mtime)

def main():
    print("=" * 70)
    print("REMEMBER TWELVE - LOCAL PHOTOS CURATION")
    print("=" * 70)

    # Find photos in 2023 Remember Twelve directory
    pictures_dir = Path.home() / 'Pictures' / '2023 Remember Twelve'
    photo_files = []

    for ext in ['*.jpg', '*.jpeg', '*.HEIC', '*.heic', '*.png', '*.JPG', '*.JPEG']:
        for photo in pictures_dir.glob(ext):
            if photo.stem.startswith('IMG_'):  # Only IMG photos
                photo_files.append(photo)

    photo_files = sorted(photo_files)  # Get all photos, no limit

    print(f"\n✓ Found {len(photo_files)} photos")

    # Analyze photos
    analyzed_photos = []
    for photo in photo_files:
        photo_date = get_photo_date(photo)

        # Basic quality scoring based on image properties
        try:
            img = Image.open(photo)
            width, height = img.size
            file_size = photo.stat().st_size / (1024 * 1024)  # MB

            # Resolution score (0-100)
            resolution_score = min(100, (width * height) / 40000)  # 4MP = 100 points

            # File size score (larger files generally have less compression)
            size_score = min(100, file_size * 30)  # 3MB+ = 100 points

            # Aspect ratio score (prefer 3:2, 4:3, 16:9)
            aspect_ratio = width / height if height > 0 else 1
            ar_score = 100 if 1.3 <= aspect_ratio <= 1.8 else 70

            quality_score = (resolution_score * 0.5 + size_score * 0.3 + ar_score * 0.2)

            # Simple emotional scoring (photos with people are more emotional)
            # This is a placeholder - in a real system you'd use ML face detection
            emotional_score = 50.0  # Base score
            if file_size > 2:  # Larger files might have more detail/people
                emotional_score += 20
            if width > 3000:  # High res photos often have more content
                emotional_score += 10

            combined_score = (quality_score * 0.6 + emotional_score * 0.4)

            analyzed_photos.append({
                'photo_path': str(photo),
                'timestamp': photo_date.isoformat(),
                'month': photo_date.month,
                'quality_score': round(quality_score, 1),
                'emotional_score': round(emotional_score, 1),
                'combined_score': round(combined_score, 1),
                'metadata': {
                    'quality_tier': 'high' if quality_score >= 70 else 'medium' if quality_score >= 50 else 'low',
                    'emotional_tier': 'high' if emotional_score >= 70 else 'acceptable' if emotional_score >= 50 else 'low',
                    'has_faces': False,
                    'face_count': 0
                }
            })
        except Exception as e:
            print(f"Warning: Could not analyze {photo.name}: {e}")
            # Fallback to defaults
            analyzed_photos.append({
                'photo_path': str(photo),
                'timestamp': photo_date.isoformat(),
                'month': photo_date.month,
                'quality_score': 50.0,
                'emotional_score': 50.0,
                'combined_score': 50.0,
                'metadata': {
                    'quality_tier': 'medium',
                    'emotional_tier': 'acceptable',
                    'has_faces': False,
                    'face_count': 0
                }
            })

    print(f"✓ Analyzed {len(analyzed_photos)} photos")

    # Create curator and distribute to 12 months
    curator = TwelveCurator(CurationConfig(strategy='balanced'))

    # Use flexible distribution
    month_distribution = curator.distribute_to_twelve_months(
        analyzed_photos,
        flexible=True
    )

    # Count filled months
    filled_months = sum(1 for photo in month_distribution.values() if photo is not None)
    print(f"✓ Distributed photos across {filled_months} months")

    # Group ALL photos by their capture month for swapping
    photos_by_month = {}
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    for photo in analyzed_photos:
        month_name = month_names[photo['month'] - 1]
        if month_name not in photos_by_month:
            photos_by_month[month_name] = []
        photos_by_month[month_name].append(photo)

    # Sort each month's photos by combined_score (best first)
    for month_name in photos_by_month:
        photos_by_month[month_name].sort(key=lambda p: p['combined_score'], reverse=True)

    # Create output
    results = {
        'year': 2023,  # These are 2023 photos
        'strategy': 'balanced',
        'created_at': datetime.now().isoformat(),
        'stats': {
            'total_candidates': len(analyzed_photos),
            'months_represented': filled_months
        },
        'photos': analyzed_photos,
        'month_distribution': month_distribution,
        'photos_by_capture_month': photos_by_month,  # NEW: All candidates grouped by month
        'user_swaps': {}  # NEW: Track manual photo selections
    }

    # Save to UI directory
    output_file = Path('ui/photos_data.json')
    output_file.write_text(json.dumps(results, indent=2))
    print(f"✓ Saved curation to {output_file}")

    # Copy photos to UI directory
    photos_dir = Path('ui/photos')
    photos_dir.mkdir(exist_ok=True)

    for i, photo_data in enumerate(analyzed_photos):
        src = Path(photo_data['photo_path'])
        # src.suffix already includes the dot, so don't add another one
        dst = photos_dir / f"{i}{src.suffix.lower()}"

        # Copy file
        import shutil
        shutil.copy2(src, dst)

    print(f"✓ Copied {len(analyzed_photos)} photos to ui/photos/")
    print("\n" + "=" * 70)
    print("✅ CURATION COMPLETE!")
    print("=" * 70)
    print(f"\nOpen http://localhost:8080/viewer_dynamic.html to see your photos!")

if __name__ == '__main__':
    main()
