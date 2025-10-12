#!/usr/bin/env python3
"""
Remember Twelve - Combined Quality + Emotional Analysis Demo

Showcases both analyzers working together:
- Feature 1.1: Photo Quality Analyzer (sharpness + exposure)
- Feature 1.2: Emotional Significance Detector (faces + emotions)

Usage:
    # Analyze a single photo
    python demo_combined_analysis.py photo.jpg

    # Analyze a directory
    python demo_combined_analysis.py ~/Photos --limit 20

    # Show detailed breakdown
    python demo_combined_analysis.py ~/Photos --limit 10 --detailed
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from photo_quality_analyzer import PhotoQualityAnalyzer, QualityScore
from emotional_significance import EmotionalAnalyzer, EmotionalScore


def print_header(title: str):
    """Print formatted header."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def print_visual_bar(score: float, width: int = 30) -> str:
    """Generate visual score bar with emojis."""
    filled = int(width * score / 100)
    empty = width - filled

    if score >= 70:
        color = "🟩"
    elif score >= 50:
        color = "🟨"
    else:
        color = "🟥"

    bar = color * filled + "⬜" * empty
    return f"[{bar}] {score:5.1f}"


def analyze_single_photo(photo_path: str, detailed: bool = False):
    """Analyze a single photo with both systems."""
    print_header("📸 Combined Photo Analysis")

    photo_name = Path(photo_path).name
    print(f"Photo: {photo_name}")
    print(f"Path:  {photo_path}")
    print()

    # Initialize analyzers
    quality_analyzer = PhotoQualityAnalyzer()
    emotional_analyzer = EmotionalAnalyzer()

    try:
        # Analyze quality
        start = time.time()
        quality_score = quality_analyzer.analyze_photo(photo_path)
        quality_time = (time.time() - start) * 1000

        # Analyze emotional
        start = time.time()
        emotional_score = emotional_analyzer.analyze_photo(photo_path)
        emotional_time = (time.time() - start) * 1000

        # Display results
        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│ QUALITY ANALYSIS (Technical Excellence)                        │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print(f"│ Sharpness:  {print_visual_bar(quality_score.sharpness):54s} │")
        print(f"│ Exposure:   {print_visual_bar(quality_score.exposure):54s} │")
        print(f"│ Composite:  {print_visual_bar(quality_score.composite):54s} │")
        print(f"│ Tier:       {quality_score.tier.upper():49s} │")
        print(f"│ Time:       {quality_time:5.1f}ms{' ' * 45} │")
        print("└─────────────────────────────────────────────────────────────────┘")
        print()

        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│ EMOTIONAL ANALYSIS (Human Significance)                        │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print(f"│ Faces:      {emotional_score.face_count} detected ({emotional_score.face_coverage*100:4.1f}% coverage){' ' * 27} │")
        print(f"│ Emotion:    {print_visual_bar(emotional_score.emotion_score):54s} │")
        print(f"│ Intimacy:   {print_visual_bar(emotional_score.intimacy_score):54s} │")
        print(f"│ Engagement: {print_visual_bar(emotional_score.engagement_score):54s} │")
        print(f"│ Composite:  {print_visual_bar(emotional_score.composite):54s} │")
        print(f"│ Tier:       {emotional_score.tier.upper():49s} │")
        print(f"│ Time:       {emotional_time:5.1f}ms{' ' * 45} │")
        print("└─────────────────────────────────────────────────────────────────┘")
        print()

        # Calculate combined score (weighted)
        combined_score = (quality_score.composite * 0.4) + (emotional_score.composite * 0.6)

        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│ COMBINED SCORE (40% Quality + 60% Emotional)                   │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print(f"│ Final Score: {print_visual_bar(combined_score):53s} │")

        # Determine tier
        if combined_score >= 70:
            tier = "HIGH"
            recommendation = "✅ EXCELLENT - Perfect for Twelve curation!"
        elif combined_score >= 50:
            tier = "MEDIUM"
            recommendation = "⚠️  GOOD - Consider including in Twelve"
        else:
            tier = "LOW"
            recommendation = "❌ SKIP - Not ideal for curation"

        print(f"│ Tier:       {tier:49s} │")
        print(f"│ Total Time: {quality_time + emotional_time:5.1f}ms{' ' * 45} │")
        print("└─────────────────────────────────────────────────────────────────┘")
        print()

        print("💡 Recommendation:")
        print(f"   {recommendation}")
        print()

        # Detailed breakdown
        if detailed:
            print("📊 Detailed Breakdown:")
            print()
            print(f"Quality Analysis:")
            print(f"  • Sharpness: {quality_score.sharpness:.1f}/100 (60% weight)")
            print(f"  • Exposure:  {quality_score.exposure:.1f}/100 (40% weight)")
            print(f"  • Formula:   ({quality_score.sharpness:.1f} × 0.6) + ({quality_score.exposure:.1f} × 0.4) = {quality_score.composite:.1f}")
            print()
            print(f"Emotional Analysis:")
            print(f"  • Face Presence: {emotional_score.face_count} face(s) = {_calculate_face_points(emotional_score.face_count)} points")
            print(f"  • Emotion Score: {emotional_score.emotion_score:.1f}/100")
            print(f"  • Intimacy:      {emotional_score.intimacy_score:.1f}/100")
            print(f"  • Engagement:    {emotional_score.engagement_score:.1f}/100")
            print()
            print(f"Combined Score:")
            print(f"  • Quality:   {quality_score.composite:.1f} × 0.4 = {quality_score.composite * 0.4:.1f}")
            print(f"  • Emotional: {emotional_score.composite:.1f} × 0.6 = {emotional_score.composite * 0.6:.1f}")
            print(f"  • Total:     {combined_score:.1f}/100")
            print()

        return True

    except Exception as e:
        print(f"❌ Error analyzing photo: {e}")
        import traceback
        traceback.print_exc()
        return False


def _calculate_face_points(face_count: int) -> int:
    """Calculate face presence points."""
    if face_count == 0:
        return 0
    elif face_count == 1:
        return 20
    elif face_count == 2:
        return 25
    else:
        return 30


def analyze_directory(directory: str, limit: int = 20, detailed: bool = False):
    """Analyze multiple photos in a directory."""
    print_header("📸 Combined Library Analysis")

    # Find photos
    photo_dir = Path(directory)
    extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    photos = []

    for ext in extensions:
        photos.extend(photo_dir.glob(f'*{ext}'))
        photos.extend(photo_dir.glob(f'*{ext.upper()}'))

    photos = sorted(photos)[:limit]

    if not photos:
        print("❌ No photos found in directory")
        return

    print(f"Found {len(photos)} photos to analyze")
    print()

    # Initialize analyzers
    quality_analyzer = PhotoQualityAnalyzer()
    emotional_analyzer = EmotionalAnalyzer()

    # Analyze all photos
    results = []

    print("Analyzing...")
    for i, photo_path in enumerate(photos, 1):
        try:
            # Quality analysis
            quality_score = quality_analyzer.analyze_photo(str(photo_path))

            # Emotional analysis
            emotional_score = emotional_analyzer.analyze_photo(str(photo_path))

            # Combined score
            combined = (quality_score.composite * 0.4) + (emotional_score.composite * 0.6)

            results.append({
                'name': photo_path.name,
                'quality': quality_score,
                'emotional': emotional_score,
                'combined': combined
            })

            # Progress
            percent = (i / len(photos)) * 100
            filled = int(40 * i / len(photos))
            bar = "█" * filled + "░" * (40 - filled)
            print(f"\r  [{bar}] {percent:5.1f}% ({i}/{len(photos)})", end="")

        except Exception as e:
            print(f"\n  ⚠️  Failed: {photo_path.name} - {e}")

    print()
    print()

    # Sort by combined score
    results.sort(key=lambda x: x['combined'], reverse=True)

    # Statistics
    avg_quality = sum(r['quality'].composite for r in results) / len(results)
    avg_emotional = sum(r['emotional'].composite for r in results) / len(results)
    avg_combined = sum(r['combined'] for r in results) / len(results)

    high_count = sum(1 for r in results if r['combined'] >= 70)
    medium_count = sum(1 for r in results if 50 <= r['combined'] < 70)
    low_count = sum(1 for r in results if r['combined'] < 50)

    with_faces = sum(1 for r in results if r['emotional'].face_count > 0)

    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│ LIBRARY STATISTICS                                              │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Total Photos:       {len(results):4d}{' ' * 41} │")
    print(f"│ Avg Quality Score:  {avg_quality:5.1f}/100{' ' * 38} │")
    print(f"│ Avg Emotional Score:{avg_emotional:5.1f}/100{' ' * 38} │")
    print(f"│ Avg Combined Score: {avg_combined:5.1f}/100{' ' * 38} │")
    print("│                                                                 │")
    print(f"│ ✅ High (70-100):    {high_count:4d}  ({high_count/len(results)*100:5.1f}%){' ' * 35} │")
    print(f"│ ⚠️  Medium (50-69):   {medium_count:4d}  ({medium_count/len(results)*100:5.1f}%){' ' * 35} │")
    print(f"│ ❌ Low (0-49):       {low_count:4d}  ({low_count/len(results)*100:5.1f}%){' ' * 35} │")
    print("│                                                                 │")
    print(f"│ 🧑 Photos with faces: {with_faces:4d}  ({with_faces/len(results)*100:5.1f}%){' ' * 34} │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()

    # Top 10
    print("🏆 Top 10 Photos for Twelve Curation:")
    print()
    for i, result in enumerate(results[:10], 1):
        q = result['quality'].composite
        e = result['emotional'].composite
        c = result['combined']
        name = result['name'][:35]

        # Tier emoji
        if c >= 70:
            emoji = "✅"
        elif c >= 50:
            emoji = "⚠️"
        else:
            emoji = "❌"

        print(f"  {i:2d}. {emoji} {name:35s}  Combined: {c:5.1f}  (Q:{q:5.1f} E:{e:5.1f})")

    print()

    # Bottom 5 if more than 15 photos
    if len(results) > 15:
        print("📉 Bottom 5 Photos:")
        print()
        for i, result in enumerate(results[-5:], 1):
            q = result['quality'].composite
            e = result['emotional'].composite
            c = result['combined']
            name = result['name'][:35]

            print(f"  {i}. ❌ {name:35s}  Combined: {c:5.1f}  (Q:{q:5.1f} E:{e:5.1f})")

        print()

    # Recommendation
    print("💡 Recommendation:")
    print(f"   {high_count} photo(s) ready for Twelve curation")
    if high_count < 12:
        print(f"   💭 Consider including {min(12 - high_count, medium_count)} from medium tier")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Remember Twelve - Combined Quality + Emotional Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg                    # Analyze single photo
  %(prog)s ~/Photos                     # Analyze directory (20 photos)
  %(prog)s ~/Photos --limit 50          # Analyze 50 photos
  %(prog)s photo.jpg --detailed         # Show detailed breakdown
        """
    )

    parser.add_argument(
        "path",
        help="Photo file or directory to analyze"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of photos to analyze (default: 20)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis breakdown"
    )

    args = parser.parse_args()

    path = Path(args.path).expanduser()

    if not path.exists():
        print(f"❌ Error: Path not found: {path}")
        sys.exit(1)

    # Single photo or directory?
    if path.is_file():
        success = analyze_single_photo(str(path), args.detailed)
        sys.exit(0 if success else 1)
    elif path.is_dir():
        analyze_directory(str(path), limit=args.limit, detailed=args.detailed)
    else:
        print(f"❌ Error: Invalid path: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
