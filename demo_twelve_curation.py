#!/usr/bin/env python3
"""
Remember Twelve - Curation Engine Demo

Demonstrates the core curation algorithm that selects the perfect 12 photos
from a year's worth of photos.

Usage:
    python demo_twelve_curation.py ~/Photos/2024
    python demo_twelve_curation.py ~/Desktop --year 2025
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from twelve_curator import TwelveCurator, CurationConfig


def print_header(title: str):
    """Print formatted header."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def demo_curation(directory: str, year: int, strategy: str = "balanced"):
    """Run curation demo on a directory."""
    print_header("📸 Twelve Curation Engine Demo")

    photo_dir = Path(directory)
    if not photo_dir.exists():
        print(f"❌ Directory not found: {directory}")
        return

    # Find all photos
    extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    photos = []

    print(f"Scanning directory: {directory}")
    for ext in extensions:
        photos.extend(photo_dir.glob(f'*{ext}'))
        photos.extend(photo_dir.glob(f'*{ext.upper()}'))

    if not photos:
        print("❌ No photos found in directory")
        return

    print(f"Found {len(photos)} photos")
    print()

    # Initialize curator with strategy
    configs = {
        'balanced': CurationConfig.balanced(),
        'people_first': CurationConfig.people_first(),
        'aesthetic_first': CurationConfig.aesthetic_first(),
        'top_heavy': CurationConfig.top_heavy()
    }

    config = configs.get(strategy, CurationConfig.balanced())
    curator = TwelveCurator(config)

    print(f"🎯 Strategy: {strategy.upper()}")
    print(f"   Quality weight: {config.quality_weight*100:.0f}%")
    print(f"   Emotional weight: {config.emotional_weight*100:.0f}%")
    print(f"   Temporal distribution: {'Enforced' if config.enforce_monthly_distribution else 'Flexible'}")
    print()

    # Curate!
    print("🔄 Analyzing photos and selecting the Twelve...")
    print()

    try:
        selection = curator.curate_year(photos, year)

        # Display results
        print_header(f"✨ Your Twelve for {year}")

        print(f"Strategy: {selection.strategy}")
        print(f"Selected: {len(selection.photos)} photos")
        print()

        print("📊 Statistics:")
        print(f"   Total candidates analyzed: {selection.stats['total_candidates']}")
        print(f"   Average quality score: {selection.stats['avg_quality']:.1f}/100")
        print(f"   Average emotional score: {selection.stats['avg_emotional']:.1f}/100")
        print(f"   Average combined score: {selection.stats['avg_combined']:.1f}/100")
        print(f"   Months represented: {selection.stats['months_represented']}/12")
        print(f"   Photos with faces: {selection.stats['photos_with_faces']}/{len(selection.photos)}")
        print()

        # Display the Twelve
        print("🏆 The Twelve:")
        print()

        for i, photo in enumerate(selection.photos, 1):
            month_name = photo.timestamp.strftime("%B") if photo.timestamp else "Unknown"
            date_str = photo.timestamp.strftime("%b %d") if photo.timestamp else "Unknown"

            # Tier emoji
            if photo.combined_score >= 70:
                emoji = "✅"
            elif photo.combined_score >= 50:
                emoji = "⚠️"
            else:
                emoji = "📷"

            name = photo.photo_path.name[:40]

            print(f"  {i:2d}. {emoji} {name:40s}")
            print(f"      Date: {date_str:10s}  |  Combined: {photo.combined_score:5.1f}  |  Q:{photo.quality_score:5.1f} E:{photo.emotional_score:5.1f}")
            print()

        # Monthly distribution
        print("📅 Monthly Distribution:")
        print()

        month_counts = {}
        for photo in selection.photos:
            if photo.month:
                month_counts[photo.month] = month_counts.get(photo.month, 0) + 1

        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        for i, month_name in enumerate(months, 1):
            count = month_counts.get(i, 0)
            bar = "█" * count + "░" * (3 - count) if count <= 3 else "█" * 3 + f"+{count-3}"
            print(f"   {month_name}: {bar}")

        print()

        # Save option
        output_path = Path(f"twelve_{year}_{strategy}.json")
        selection.save(output_path)
        print(f"💾 Saved selection to: {output_path}")
        print()

        # Recommendation
        print("💡 Recommendation:")
        avg_score = selection.stats['avg_combined']
        if avg_score >= 70:
            print(f"   ✅ Excellent curation! Average score: {avg_score:.1f}/100")
        elif avg_score >= 50:
            print(f"   ⚠️  Good curation. Average score: {avg_score:.1f}/100")
        else:
            print(f"   📷 Photos selected. Consider importing higher quality photos.")

        print()

        return selection

    except Exception as e:
        print(f"❌ Error during curation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Remember Twelve - Twelve Curation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/Photos/2024                # Curate 2024 photos
  %(prog)s ~/Desktop --year 2025        # Curate 2025 photos from Desktop
  %(prog)s ~/Photos --strategy people_first  # People-focused strategy
  %(prog)s ~/Photos --strategy aesthetic_first  # Quality-focused strategy
        """
    )

    parser.add_argument(
        "directory",
        help="Directory containing photos to curate"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now().year,
        help="Year to curate (default: current year)"
    )
    parser.add_argument(
        "--strategy",
        choices=['balanced', 'people_first', 'aesthetic_first', 'top_heavy'],
        default='balanced',
        help="Curation strategy (default: balanced)"
    )

    args = parser.parse_args()

    demo_curation(args.directory, args.year, args.strategy)


if __name__ == "__main__":
    main()
