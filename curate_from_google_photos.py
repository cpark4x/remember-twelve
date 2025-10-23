#!/usr/bin/env python3
"""
Remember Twelve - Curate from Google Photos

Complete end-to-end demo that:
1. Authenticates with Google Photos
2. Fetches photos from a specific year
3. Analyzes quality + emotional significance
4. Curates the best 12 photos
5. Saves results with Google Photos links

This is the complete Feature 1.5 (Google Photos Integration) working end-to-end!

Prerequisites:
1. Google Cloud project with Photos Library API enabled
2. OAuth credentials as google_photos_credentials.json
3. Dependencies: pip install -r requirements.txt

Usage:
    # Curate photos from 2023
    python curate_from_google_photos.py --year 2023

    # With custom strategy
    python curate_from_google_photos.py --year 2024 --strategy people_first

    # Limit number of photos to analyze (for testing)
    python curate_from_google_photos.py --year 2023 --limit 50
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from photo_sources import PhotoSourceFactory
from twelve_curator import TwelveCurator
from twelve_curator.data_classes import CurationConfig


def main():
    """Run complete Google Photos curation pipeline."""
    parser = argparse.ArgumentParser(
        description="Curate 12 best photos from Google Photos"
    )
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Year to curate (e.g., 2023)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='balanced',
        choices=['balanced', 'aesthetic_first', 'people_first', 'top_heavy'],
        help='Curation strategy (default: balanced)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of photos to analyze (for testing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file (default: twelve_{year}_{strategy}.json)'
    )
    parser.add_argument(
        '--credentials',
        type=str,
        default='google_photos_credentials.json',
        help='Path to OAuth credentials (default: google_photos_credentials.json)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("REMEMBER TWELVE - Curate from Google Photos")
    print("=" * 70)
    print()
    print(f"Year: {args.year}")
    print(f"Strategy: {args.strategy}")
    print()

    # Check for credentials
    creds_path = Path(args.credentials)
    if not creds_path.exists():
        print(f"ERROR: {args.credentials} not found!")
        print()
        print("To get credentials:")
        print("1. Go to https://console.cloud.google.com")
        print("2. Create a project and enable Google Photos Library API")
        print("3. Create OAuth 2.0 credentials (Desktop app)")
        print("4. Download as google_photos_credentials.json")
        print()
        return 1

    # Step 1: Create Google Photos source
    print("Step 1: Creating Google Photos source...")
    try:
        source = PhotoSourceFactory.create_google_photos(str(creds_path))
        print("✓ Source created")
        print()
    except Exception as e:
        print(f"✗ Failed to create source: {e}")
        return 1

    # Step 2: Authenticate
    print("Step 2: Authenticating with Google Photos...")
    print("(Browser will open for OAuth consent if not already authenticated)")
    print()

    try:
        user_email = source.authenticate()
        print(f"✓ Authenticated as: {user_email}")
        print()
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return 1

    # Step 3: Create curator with strategy
    print(f"Step 3: Initializing curator (strategy: {args.strategy})...")

    if args.strategy == 'balanced':
        config = CurationConfig.balanced()
    elif args.strategy == 'aesthetic_first':
        config = CurationConfig.aesthetic_first()
    elif args.strategy == 'people_first':
        config = CurationConfig.people_first()
    elif args.strategy == 'top_heavy':
        config = CurationConfig.top_heavy()

    curator = TwelveCurator(config)
    print("✓ Curator ready")
    print()

    # Step 4: Curate from Google Photos
    print(f"Step 4: Curating photos from {args.year}...")
    print("(This will download photos, analyze quality + emotions, and select best 12)")
    print()

    # Progress tracking
    photo_count = 0

    def progress_callback(current, total, msg):
        nonlocal photo_count
        photo_count = current
        if current % 10 == 0:  # Print every 10 photos
            print(f"  Processed {current} photos...")

    try:
        # Run complete curation pipeline
        start_time = datetime.now()

        selection = curator.curate_from_source(
            source,
            year=args.year,
            strategy=args.strategy,
            progress_callback=progress_callback
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        print()
        print(f"✓ Curation complete!")
        print(f"  Analyzed {photo_count} photos in {elapsed:.1f}s")
        print()

    except Exception as e:
        print(f"✗ Curation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 5: Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Selected {len(selection.photos)} photos for {args.year}")
    print(f"Strategy: {selection.strategy}")
    print(f"Average combined score: {selection.stats['avg_combined']:.1f}")
    print(f"Months represented: {selection.stats['months_represented']}/12")
    print(f"Photos with faces: {selection.stats['photos_with_faces']}")
    print()

    print("Selected Photos:")
    print("-" * 70)

    for i, photo in enumerate(selection.photos, 1):
        month_str = f"Month {photo.month:02d}" if photo.month else "Unknown"
        date_str = photo.timestamp.strftime("%Y-%m-%d") if photo.timestamp else "Unknown date"

        print(f"{i:2d}. {photo.photo_path.name}")
        print(f"    Date: {date_str} ({month_str})")
        print(f"    Scores: Combined={photo.combined_score:.1f}, "
              f"Quality={photo.quality_score:.1f}, "
              f"Emotional={photo.emotional_score:.1f}")

        # Google Photos URL
        url = photo.metadata.get('original_url')
        if url:
            print(f"    Google Photos: {url}")

        print()

    # Step 6: Save results
    output_file = args.output or f"twelve_{args.year}_{args.strategy}.json"
    output_path = Path(output_file)

    print("=" * 70)
    print(f"Saving results to {output_file}...")

    try:
        selection.save(output_path)
        print(f"✓ Saved to {output_path}")
        print()
    except Exception as e:
        print(f"✗ Failed to save: {e}")
        return 1

    # Summary
    print("=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print()
    print("What's next:")
    print(f"1. View results: cat {output_file}")
    print(f"2. Open in viewer: open ui/viewer.html")
    print(f"3. Click Google Photos links to view originals")
    print()
    print("Feature 1.5 (Google Photos Integration) is complete! 🎉")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
