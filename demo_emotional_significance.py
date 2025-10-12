#!/usr/bin/env python3
"""
Emotional Significance Detector - Demo Script

This script demonstrates the capabilities of the Emotional Significance Detector
by analyzing the test fixture images and showing detailed results.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.emotional_significance import (
    EmotionalAnalyzer,
    get_default_config,
    get_emotion_focused_config,
    get_intimacy_focused_config
)


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_score_details(score, photo_name):
    """Print detailed score information."""
    print(f"\n📷 Photo: {photo_name}")
    print("-" * 70)
    print(f"  Faces Detected:    {score.face_count}")
    print(f"  Face Coverage:     {score.face_coverage * 100:.1f}%")
    print(f"  Emotion Score:     {score.emotion_score:.1f}/100")
    print(f"  Intimacy Score:    {score.intimacy_score:.1f}/100")
    print(f"  Engagement Score:  {score.engagement_score:.1f}/100")
    print(f"  Composite Score:   {score.composite:.1f}/100")
    print(f"  Significance Tier: {score.tier.upper()}")

    # Processing time
    if 'processing_time_ms' in score.metadata:
        print(f"  Processing Time:   {score.metadata['processing_time_ms']:.1f}ms")

    # Additional details
    if score.has_faces:
        num_smiling = score.metadata.get('num_smiling', 0)
        print(f"\n  Details:")
        print(f"    • {num_smiling} out of {score.face_count} faces smiling")

        if 'engagement_analysis' in score.metadata:
            eng = score.metadata['engagement_analysis']
            frontal = eng.get('frontal_count', 0)
            print(f"    • {frontal} out of {score.face_count} faces facing camera")

        if 'intimacy_analysis' in score.metadata and score.has_multiple_people:
            intim = score.metadata['intimacy_analysis']
            print(f"    • Physical closeness: {intim.get('analysis', 'N/A')}")

    # Recommendation
    print(f"\n  Recommendation:")
    if score.is_high_significance:
        print(f"    ✓ HIGH significance - Prioritize for curation")
    elif score.is_medium_significance:
        print(f"    • MEDIUM significance - Include if space allows")
    else:
        print(f"    ✗ LOW significance - May exclude from curation")


def demo_basic_analysis():
    """Demonstrate basic photo analysis."""
    print_header("Demo 1: Basic Photo Analysis")

    analyzer = EmotionalAnalyzer()
    fixtures_dir = Path(__file__).parent / 'tests' / 'emotional_significance' / 'fixtures'

    test_photos = [
        ('Single Face Portrait', 'single_face.jpg'),
        ('Couple Photo', 'couple.jpg'),
        ('Group Photo', 'group.jpg'),
        ('Landscape (No Faces)', 'landscape.jpg')
    ]

    for name, filename in test_photos:
        photo_path = fixtures_dir / filename
        if photo_path.exists():
            score = analyzer.analyze_photo(str(photo_path))
            print_score_details(score, name)
        else:
            print(f"\n⚠️  {name}: File not found")


def demo_batch_analysis():
    """Demonstrate batch analysis."""
    print_header("Demo 2: Batch Analysis")

    analyzer = EmotionalAnalyzer()
    fixtures_dir = Path(__file__).parent / 'tests' / 'emotional_significance' / 'fixtures'

    photos = list(fixtures_dir.glob('*.jpg'))
    if not photos:
        print("No test photos found")
        return

    print(f"\nAnalyzing {len(photos)} photos...")

    scores = analyzer.analyze_batch([str(p) for p in photos])

    # Filter and rank
    valid_scores = [(p, s) for p, s in zip(photos, scores) if s]

    print(f"\n✓ Successfully analyzed: {len(valid_scores)}/{len(photos)}")

    # Show summary
    print("\nResults Summary:")
    print("-" * 70)

    for photo, score in sorted(valid_scores, key=lambda x: x[1].composite, reverse=True):
        tier_symbol = "🌟" if score.tier == 'high' else "⭐" if score.tier == 'medium' else "・"
        print(f"  {tier_symbol} {photo.name:25s} - Score: {score.composite:5.1f} ({score.tier})")


def demo_custom_configs():
    """Demonstrate different configurations."""
    print_header("Demo 3: Configuration Presets")

    fixtures_dir = Path(__file__).parent / 'tests' / 'emotional_significance' / 'fixtures'
    test_photo = fixtures_dir / 'couple.jpg'

    if not test_photo.exists():
        print("Test photo not found")
        return

    configs = [
        ("Default Configuration", get_default_config()),
        ("Emotion-Focused", get_emotion_focused_config()),
        ("Intimacy-Focused", get_intimacy_focused_config()),
    ]

    for config_name, config in configs:
        analyzer = EmotionalAnalyzer(config=config)
        score = analyzer.analyze_photo(str(test_photo))

        print(f"\n{config_name}:")
        print(f"  Composite: {score.composite:.1f}")
        print(f"  Tier: {score.tier}")
        print(f"  Emotion Weight: {config.scoring_weights.emotion_weight:.0f} points")
        print(f"  Intimacy Weight: {config.scoring_weights.intimacy_weight:.0f} points")


def demo_statistics():
    """Demonstrate scoring statistics."""
    print_header("Demo 4: Score Statistics")

    from src.emotional_significance.scoring.composite import get_statistics

    analyzer = EmotionalAnalyzer()
    fixtures_dir = Path(__file__).parent / 'tests' / 'emotional_significance' / 'fixtures'

    photos = list(fixtures_dir.glob('*.jpg'))
    if not photos:
        print("No test photos found")
        return

    scores = [s for s in analyzer.analyze_batch([str(p) for p in photos]) if s]

    stats = get_statistics(scores)

    print(f"\nLibrary Statistics:")
    print("-" * 70)
    print(f"  Total Photos:      {stats['count']}")
    print(f"  Average Score:     {stats['avg_composite']:.1f}/100")
    print(f"  Highest Score:     {stats['max_composite']:.1f}/100")
    print(f"  Lowest Score:      {stats['min_composite']:.1f}/100")
    print(f"\n  Tier Distribution:")
    print(f"    High:    {stats['tier_distribution']['high']} photos")
    print(f"    Medium:  {stats['tier_distribution']['medium']} photos")
    print(f"    Low:     {stats['tier_distribution']['low']} photos")
    print(f"\n  Face Detection:")
    print(f"    Photos with faces: {stats['photos_with_faces']}")
    print(f"    Average faces:     {stats['avg_face_count']:.1f}")


def main():
    """Run all demos."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Emotional Significance Detector - Demo".center(68) + "║")
    print("║" + "  Phase 1: Core Detection (MVP)".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    try:
        demo_basic_analysis()
        demo_batch_analysis()
        demo_custom_configs()
        demo_statistics()

        print_header("Demo Complete")
        print("\n✅ All demos executed successfully!")
        print("\nTo use in your code:")
        print("  from emotional_significance import EmotionalAnalyzer")
        print("  analyzer = EmotionalAnalyzer()")
        print("  score = analyzer.analyze_photo('your_photo.jpg')")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
