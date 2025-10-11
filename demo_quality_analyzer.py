#!/usr/bin/env python3
"""
Remember Twelve - Photo Quality Analyzer Demo

Unified demo showcasing both Phase 1 (Core Algorithm) and Phase 2 (Infrastructure).

Features:
- Analyze single photos or entire directories
- Batch processing with parallel workers
- Intelligent caching (reuses previous results)
- Performance monitoring
- Quality distribution visualization
- Export results to JSON

Usage:
    # Analyze a single photo
    python demo_quality_analyzer.py photo.jpg

    # Analyze a directory (first 100 photos)
    python demo_quality_analyzer.py ~/Photos

    # Analyze with custom limit
    python demo_quality_analyzer.py ~/Photos --limit 500

    # Force re-analysis (ignore cache)
    python demo_quality_analyzer.py ~/Photos --no-cache

    # Export results to JSON
    python demo_quality_analyzer.py ~/Photos --export results.json

    # Show detailed per-photo scores
    python demo_quality_analyzer.py ~/Photos --detailed
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from photo_quality_analyzer import (
    PhotoQualityAnalyzer,
    QualityScore,
    LibraryScanner,
    BatchProcessor,
    ResultCache,
    PerformanceMonitor,
)


def print_header(title: str):
    """Print a formatted header."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def print_section(title: str):
    """Print a section header."""
    print(f"\n{title}")
    print("─" * 70)


def print_quality_bar(score: float, width: int = 30) -> str:
    """Generate a visual quality bar."""
    filled = int(width * score / 100)
    empty = width - filled

    if score >= 70:
        color = "🟩"
        tier_emoji = "✅"
    elif score >= 50:
        color = "🟨"
        tier_emoji = "⚠️"
    else:
        color = "🟥"
        tier_emoji = "❌"

    bar = color * filled + "⬜" * empty
    return f"{tier_emoji} [{bar}] {score:5.1f}"


def analyze_single_photo(photo_path: str, detailed: bool = False):
    """Analyze a single photo and display results."""
    print_header("📸 Photo Quality Analysis")

    print(f"Photo: {Path(photo_path).name}")
    print(f"Path:  {photo_path}")
    print()

    # Analyze
    analyzer = PhotoQualityAnalyzer()

    try:
        with PerformanceMonitor() as monitor:
            score = analyzer.analyze_photo(photo_path)
            monitor.record_photo()

        # Display results
        print_section("Quality Scores")
        print(f"  Sharpness:  {print_quality_bar(score.sharpness)}")
        print(f"  Exposure:   {print_quality_bar(score.exposure)}")
        print(f"  Composite:  {print_quality_bar(score.composite)}")
        print()
        print(f"  Overall Tier: {score.tier.upper()}")

        # Detailed breakdown
        if detailed:
            print_section("Detailed Analysis")
            print(f"  Algorithm: Laplacian Variance (sharpness) + Histogram (exposure)")
            print(f"  Weights: Sharpness 60%, Exposure 40%")
            print(f"  Formula: ({score.sharpness:.1f} × 0.6) + ({score.exposure:.1f} × 0.4) = {score.composite:.1f}")

        # Performance
        print_section("Performance")
        monitor.print_summary()

        # Recommendation
        print_section("Recommendation")
        if score.tier == "high":
            print("  ✅ Excellent quality! Perfect for Twelve curation.")
        elif score.tier == "acceptable":
            print("  ⚠️  Acceptable quality. May include if needed for diversity.")
        else:
            print("  ❌ Low quality. Consider excluding from curation.")

        return True

    except Exception as e:
        print(f"❌ Error analyzing photo: {e}")
        return False


def analyze_directory(
    directory: str,
    limit: int = 100,
    use_cache: bool = True,
    detailed: bool = False,
    export_path: Optional[str] = None
):
    """Analyze all photos in a directory."""
    print_header(f"📸 Photo Library Analysis")

    # Step 1: Scan
    print("🔍 Step 1: Scanning library...")
    scanner = LibraryScanner()
    photos = scanner.scan_and_collect(directory, limit=limit)

    stats = scanner.get_stats()
    print(f"   Found: {stats.total_found} photos")
    print(f"   Skipped: {stats.skipped} files")
    if stats.errors > 0:
        print(f"   Errors: {stats.errors} files")

    if not photos:
        print("\n❌ No photos found.")
        return

    # Step 2: Cache setup
    cache_path = Path(tempfile.gettempdir()) / "remember_twelve_cache.db"
    cache = ResultCache(str(cache_path)) if use_cache else None

    if use_cache:
        print(f"\n💾 Step 2: Cache check...")
        print(f"   Location: {cache_path}")

        photos_to_analyze = []
        cached_scores = []

        for photo in photos:
            if cache.should_analyze(photo):
                photos_to_analyze.append(photo)
            else:
                cached_scores.append((photo, cache.get(photo)))

        print(f"   Cached: {len(cached_scores)} photos")
        print(f"   To analyze: {len(photos_to_analyze)} photos")
    else:
        photos_to_analyze = photos
        cached_scores = []

    # Step 3: Batch processing
    all_results = []

    if photos_to_analyze:
        print(f"\n⚡ Step 3: Analyzing photos...")
        print(f"   Workers: 4 parallel processes")
        print(f"   Photos: {len(photos_to_analyze)}")
        print()

        processor = BatchProcessor(num_workers=4)

        def show_progress(current: int, total: int, failed: int = 0):
            percent = current / total * 100
            bar_length = 40
            filled = int(bar_length * current / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            status = f"{current}/{total}"
            if failed > 0:
                status += f" ({failed} failed)"
            print(f"\r   [{bar}] {percent:5.1f}% {status}", end="")

        with PerformanceMonitor() as monitor:
            result = processor.process_batch(
                photos_to_analyze,
                progress_callback=show_progress
            )
            monitor.record_photo(result.successful)

        print()  # New line

        # Cache new results
        if use_cache:
            for photo_path, score in result.scores:
                cache.set(photo_path, score)

        # Combine with cached
        all_results = [(Path(p).name, s) for p, s in result.scores]
        all_results.extend([(Path(p).name, s) for p, s in cached_scores])

        # Performance summary
        print_section("Performance Metrics")
        monitor.print_summary()

        if result.errors and detailed:
            print(f"\n   ⚠️  {len(result.errors)} photos failed:")
            for photo, error in result.errors[:5]:
                print(f"      - {Path(photo).name}: {str(error)[:60]}")
    else:
        print(f"\n✅ All results cached! No analysis needed.")
        all_results = [(Path(p).name, s) for p, s in cached_scores]

    # Step 4: Results analysis
    print_section("Quality Distribution")

    high_quality = [r for r in all_results if r[1].tier == "high"]
    acceptable = [r for r in all_results if r[1].tier == "acceptable"]
    low_quality = [r for r in all_results if r[1].tier == "low"]

    total = len(all_results)
    print(f"\n   Total: {total} photos")
    print(f"   ✅ High quality (70-100):    {len(high_quality):4d}  ({len(high_quality)/total*100:5.1f}%)")
    print(f"   ⚠️  Acceptable (50-69):       {len(acceptable):4d}  ({len(acceptable)/total*100:5.1f}%)")
    print(f"   ❌ Low quality (0-49):        {len(low_quality):4d}  ({len(low_quality)/total*100:5.1f}%)")

    # Visual distribution
    print("\n   Visual Distribution:")
    high_bar = "🟩" * int(len(high_quality) / total * 50)
    accept_bar = "🟨" * int(len(acceptable) / total * 50)
    low_bar = "🟥" * int(len(low_quality) / total * 50)
    print(f"   {high_bar}{accept_bar}{low_bar}")

    # Top and bottom photos
    all_results.sort(key=lambda x: x[1].composite, reverse=True)

    print_section("Top 5 Photos")
    for i, (name, score) in enumerate(all_results[:5], 1):
        print(f"   {i}. {name[:50]:<50}  {score.composite:5.1f}")

    if len(all_results) > 10:
        print_section("Bottom 5 Photos")
        for i, (name, score) in enumerate(all_results[-5:], 1):
            print(f"   {i}. {name[:50]:<50}  {score.composite:5.1f}")

    # Detailed listing
    if detailed and len(all_results) <= 20:
        print_section("All Photos (Detailed)")
        for name, score in all_results:
            tier_symbol = {"high": "✅", "acceptable": "⚠️", "low": "❌"}[score.tier]
            print(f"   {tier_symbol} {score.composite:5.1f} | S:{score.sharpness:5.1f} E:{score.exposure:5.1f} | {name}")

    # Cache stats
    if use_cache:
        cache_stats = cache.get_stats()
        print_section("Cache Statistics")
        print(f"   Total entries: {cache_stats['total_entries']}")
        print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"   Location: {cache_path}")

    # Recommendation
    print_section("💡 Recommendation")
    print(f"   {len(high_quality)} photos ready for Twelve curation")
    if len(high_quality) < 12:
        print(f"   💭 Consider including {12 - len(high_quality)} from acceptable tier")

    # Export
    if export_path:
        export_results(all_results, export_path)
        print(f"\n   📄 Results exported to: {export_path}")

    print()


def export_results(results: List[Tuple[str, QualityScore]], path: str):
    """Export results to JSON."""
    data = {
        "total": len(results),
        "summary": {
            "high": len([r for r in results if r[1].tier == "high"]),
            "acceptable": len([r for r in results if r[1].tier == "acceptable"]),
            "low": len([r for r in results if r[1].tier == "low"]),
        },
        "photos": [
            {
                "name": name,
                "composite": score.composite,
                "sharpness": score.sharpness,
                "exposure": score.exposure,
                "tier": score.tier,
            }
            for name, score in results
        ]
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Remember Twelve - Photo Quality Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg                    # Analyze single photo
  %(prog)s ~/Photos                     # Analyze directory (100 photos)
  %(prog)s ~/Photos --limit 500         # Analyze 500 photos
  %(prog)s ~/Photos --no-cache          # Force re-analysis
  %(prog)s ~/Photos --export out.json   # Export results
  %(prog)s ~/Photos --detailed          # Show detailed scores
        """
    )

    parser.add_argument(
        "path",
        help="Photo file or directory to analyze"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of photos to analyze (default: 100)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (force re-analysis)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis and per-photo scores"
    )
    parser.add_argument(
        "--export",
        metavar="FILE",
        help="Export results to JSON file"
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
        analyze_directory(
            str(path),
            limit=args.limit,
            use_cache=not args.no_cache,
            detailed=args.detailed,
            export_path=args.export
        )
    else:
        print(f"❌ Error: Invalid path: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
