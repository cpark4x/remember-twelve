#!/usr/bin/env python3
"""
Quick demo script to test the Photo Quality Analyzer on real photos.
"""

import sys
from pathlib import Path

# Add src to path so we can import photo_quality_analyzer
sys.path.insert(0, str(Path(__file__).parent / "src"))

from photo_quality_analyzer import PhotoQualityAnalyzer

def analyze_photos(photo_paths):
    """Analyze a list of photos and display results."""

    # Initialize analyzer
    print("🔍 Initializing Photo Quality Analyzer...\n")
    analyzer = PhotoQualityAnalyzer()

    results = []

    for photo_path in photo_paths:
        path = Path(photo_path)
        if not path.exists():
            print(f"❌ File not found: {photo_path}")
            continue

        print(f"📸 Analyzing: {path.name}")

        try:
            # Analyze the photo
            score = analyzer.analyze_photo(str(path))

            # Display results
            print(f"   Sharpness:  {score.sharpness:5.1f}/100")
            print(f"   Exposure:   {score.exposure:5.1f}/100")
            print(f"   Composite:  {score.composite:5.1f}/100")
            print(f"   Quality:    {score.tier.upper()}")

            # Emoji for visual feedback
            emoji = "✅" if score.tier == "high" else "⚠️" if score.tier == "acceptable" else "❌"
            print(f"   {emoji}\n")

            results.append((path.name, score))

        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    # Summary
    if results:
        print("=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)

        # Sort by composite score (highest first)
        results.sort(key=lambda x: x[1].composite, reverse=True)

        high_quality = [r for r in results if r[1].tier == "high"]
        acceptable = [r for r in results if r[1].tier == "acceptable"]
        low_quality = [r for r in results if r[1].tier == "low"]

        print(f"\nTotal photos analyzed: {len(results)}")
        print(f"✅ High quality (70-100):      {len(high_quality)}")
        print(f"⚠️  Acceptable (50-69):         {len(acceptable)}")
        print(f"❌ Low quality (0-49):          {len(low_quality)}")

        if high_quality:
            print(f"\n🏆 Best photo: {high_quality[0][0]} ({high_quality[0][1].composite:.1f})")
        if low_quality:
            print(f"📉 Worst photo: {low_quality[-1][0]} ({low_quality[-1][1].composite:.1f})")

        # Recommendation for curation
        print(f"\n💡 Recommendation: {len(high_quality)} photo(s) suitable for Twelve curation")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_analyzer_demo.py <photo1> [photo2] [photo3] ...")
        print("\nExample:")
        print("  python test_analyzer_demo.py ~/Desktop/*.png")
        sys.exit(1)

    photo_paths = sys.argv[1:]
    analyze_photos(photo_paths)
