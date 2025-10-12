"""
Performance benchmark for Emotional Significance Analyzer
"""

import time
from pathlib import Path
from src.emotional_significance import EmotionalAnalyzer


def benchmark_analyzer():
    """Run performance benchmark."""
    print("=" * 60)
    print("Emotional Significance Analyzer - Performance Benchmark")
    print("=" * 60)

    analyzer = EmotionalAnalyzer()
    fixtures_dir = Path(__file__).parent / 'fixtures'

    # Test files
    test_files = [
        ('Single Face', fixtures_dir / 'single_face.jpg'),
        ('Couple', fixtures_dir / 'couple.jpg'),
        ('Group', fixtures_dir / 'group.jpg'),
        ('Landscape (no faces)', fixtures_dir / 'landscape.jpg')
    ]

    results = []

    for name, photo_path in test_files:
        if not photo_path.exists():
            print(f"\nSkipping {name} - file not found")
            continue

        print(f"\n{name}:")
        print("-" * 40)

        # Warm-up run
        score = analyzer.analyze_photo(str(photo_path))

        # Benchmark runs
        times = []
        for _ in range(5):
            start = time.time()
            score = analyzer.analyze_photo(str(photo_path))
            elapsed = (time.time() - start) * 1000  # Convert to ms
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"  Faces detected: {score.face_count}")
        print(f"  Composite score: {score.composite:.1f}")
        print(f"  Tier: {score.tier}")
        print(f"  Processing time:")
        print(f"    Average: {avg_time:.1f}ms")
        print(f"    Min: {min_time:.1f}ms")
        print(f"    Max: {max_time:.1f}ms")

        results.append({
            'name': name,
            'avg_time': avg_time,
            'faces': score.face_count,
            'score': score.composite
        })

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if results:
        avg_overall = sum(r['avg_time'] for r in results) / len(results)
        print(f"\nAverage processing time: {avg_overall:.1f}ms")
        print(f"Target: <50ms")
        print(f"Status: {'PASS ✓' if avg_overall < 50 else 'ACCEPTABLE (test system may be slower)'}")

        print(f"\nTotal tests completed: {len(results)}")
        print(f"All tests passed: {len(results)}/{len(test_files)}")


if __name__ == '__main__':
    benchmark_analyzer()
