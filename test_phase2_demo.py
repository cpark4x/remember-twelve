#!/usr/bin/env python3
"""
Integration demo for Photo Quality Analyzer Phase 2 Infrastructure.

This script demonstrates the complete production pipeline:
1. LibraryScanner - Discover photos in a directory
2. ResultCache - Check for previously analyzed photos
3. BatchProcessor - Analyze photos in parallel
4. PerformanceMonitor - Track performance metrics

Usage:
    python test_phase2_demo.py <directory_path>
"""

import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from photo_quality_analyzer import (
    PhotoQualityAnalyzer,
    LibraryScanner,
    BatchProcessor,
    ResultCache,
    PerformanceMonitor,
)


def demo_full_pipeline(directory: str, limit: int = 100):
    """Run complete Phase 2 pipeline demonstration."""

    print("=" * 70)
    print("📸 PHOTO QUALITY ANALYZER - PHASE 2 DEMO")
    print("=" * 70)
    print()

    # Step 1: Scan Library
    print("🔍 Step 1: Scanning photo library...")
    print(f"   Directory: {directory}")
    print()

    scanner = LibraryScanner()
    photos = scanner.scan_and_collect(directory, limit=limit)

    scan_stats = scanner.get_stats()
    print(f"   Found: {scan_stats.total_found} photos")
    print(f"   Skipped: {scan_stats.skipped} files")
    if scan_stats.errors > 0:
        print(f"   Errors: {scan_stats.errors} files")
    print()

    if not photos:
        print("   ❌ No photos found. Exiting.")
        return

    # Step 2: Initialize Cache
    print("💾 Step 2: Initializing result cache...")

    # Use temp directory for demo
    cache_path = Path(tempfile.gettempdir()) / "remember_twelve_demo.db"
    cache = ResultCache(str(cache_path))
    print(f"   Cache: {cache_path}")
    print()

    # Step 3: Check Cache
    print("🔎 Step 3: Checking cache for existing results...")

    photos_to_analyze = []
    cached_scores = []

    for photo in photos:
        cached = cache.get(photo)
        if cached:
            cached_scores.append((photo, cached))
        else:
            photos_to_analyze.append(photo)

    print(f"   Cached: {len(cached_scores)} photos (reusing results)")
    print(f"   Need analysis: {len(photos_to_analyze)} photos")
    print()

    # Step 4: Batch Processing
    all_scores = []

    if photos_to_analyze:
        print("⚡ Step 4: Batch processing with parallel workers...")
        print(f"   Workers: 4 processes")
        print(f"   Photos: {len(photos_to_analyze)}")
        print()

        processor = BatchProcessor(num_workers=4)

        # Progress callback
        def show_progress(current: int, total: int, failed: int = 0):
            percent = (current / total) * 100
            bar_length = 40
            filled = int(bar_length * current / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            status = f"{current}/{total}"
            if failed > 0:
                status += f" ({failed} failed)"
            print(f"\r   Progress: [{bar}] {percent:5.1f}% {status}", end="")

        # Process with monitoring
        with PerformanceMonitor() as monitor:
            result = processor.process_batch(
                photos_to_analyze,
                progress_callback=show_progress
            )
            monitor.record_photo(result.successful)

        print()  # New line after progress bar
        print()

        # Step 5: Cache Results
        print("💾 Step 5: Caching new results...")
        for photo_path, score in result.scores:
            cache.set(photo_path, score)
            all_scores.append((Path(photo_path).name, score))

        print(f"   Cached: {len(result.scores)} new results")

        if result.errors:
            print(f"   Errors: {len(result.errors)} photos failed")
            for photo, error in result.errors[:3]:  # Show first 3 errors
                print(f"      - {Path(photo).name}: {error}")
        print()

        # Performance Summary
        print("📊 Step 6: Performance Metrics")
        print("─" * 70)
        monitor.print_summary()
        print()
    else:
        print("✅ Step 4-6: Skipped (all results cached)")
        print()
        all_scores = [(Path(p).name, s) for p, s in cached_scores]

    # Step 7: Results Summary
    print("=" * 70)
    print("📊 ANALYSIS RESULTS")
    print("=" * 70)
    print()

    if not all_scores and cached_scores:
        all_scores = [(Path(p).name, s) for p, s in cached_scores]

    # Sort by composite score
    all_scores.sort(key=lambda x: x[1].composite, reverse=True)

    high_quality = [s for s in all_scores if s[1].tier == "high"]
    acceptable = [s for s in all_scores if s[1].tier == "acceptable"]
    low_quality = [s for s in all_scores if s[1].tier == "low"]

    print(f"Total analyzed: {len(all_scores)} photos")
    print(f"✅ High quality (70-100):    {len(high_quality):3d} ({len(high_quality)/len(all_scores)*100:.1f}%)")
    print(f"⚠️  Acceptable (50-69):       {len(acceptable):3d} ({len(acceptable)/len(all_scores)*100:.1f}%)")
    print(f"❌ Low quality (0-49):        {len(low_quality):3d} ({len(low_quality)/len(all_scores)*100:.1f}%)")
    print()

    # Top 5 and Bottom 5
    if high_quality:
        print("🏆 Top 5 Photos:")
        for i, (name, score) in enumerate(all_scores[:5], 1):
            print(f"   {i}. {name[:50]:<50} ({score.composite:.1f})")
        print()

    if len(all_scores) > 5:
        print("📉 Bottom 5 Photos:")
        for i, (name, score) in enumerate(all_scores[-5:], 1):
            print(f"   {i}. {name[:50]:<50} ({score.composite:.1f})")
        print()

    # Cache Statistics
    cache_stats = cache.get_stats()
    print("💾 Cache Statistics:")
    print(f"   Total entries: {cache_stats['total_entries']}")
    print(f"   Hits: {cache_stats['hits']}")
    print(f"   Misses: {cache_stats['misses']}")
    print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
    print()

    # Recommendation
    print("💡 Recommendation:")
    print(f"   {len(high_quality)} photo(s) ready for Twelve curation")
    print()

    print("=" * 70)
    print("✅ Phase 2 Demo Complete!")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_phase2_demo.py <directory> [limit]")
        print()
        print("Examples:")
        print("  python test_phase2_demo.py ~/Desktop")
        print("  python test_phase2_demo.py ~/Pictures 50")
        print()
        print("This will scan the directory, cache results, and analyze photos in parallel.")
        sys.exit(1)

    directory = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    demo_full_pipeline(directory, limit)


if __name__ == "__main__":
    main()
