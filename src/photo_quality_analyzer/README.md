# Photo Quality Analyzer

A modular, high-performance system for analyzing photo quality based on sharpness and exposure metrics.

**Current Version:** 2.0.0 (Phase 2 - Production Infrastructure)

## Overview

The Photo Quality Analyzer automatically scores photos on a 0-100 scale to identify high-quality images for curation. It uses proven computer vision algorithms without requiring machine learning.

**Key Metrics:**
- **Sharpness (60% weight):** Detects blur using Laplacian variance
- **Exposure (40% weight):** Analyzes over/underexposure using histogram distribution
- **Composite Score:** Weighted combination of both metrics

**Quality Tiers:**
- **High (70-100):** Prioritize for curation
- **Acceptable (50-69):** Include if needed for diversity
- **Low (0-49):** Exclude from automatic curation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individual packages
pip install opencv-python numpy Pillow pytest pytest-cov
```

**Requirements:** Python 3.11+

## What's New in Phase 2

Phase 2 adds production-ready infrastructure for analyzing large photo libraries:

- **Batch Processing:** Parallel analysis with progress tracking
- **Result Caching:** SQLite-backed cache with hash-based invalidation
- **Library Scanning:** Recursive photo discovery with filtering
- **Performance Monitoring:** Track throughput and memory usage

## Quick Start

### Basic Usage (Phase 1)

```python
from photo_quality_analyzer import PhotoQualityAnalyzer

# Initialize analyzer
analyzer = PhotoQualityAnalyzer()

# Analyze a photo
score = analyzer.analyze_photo('family_photo.jpg')

print(f"Sharpness:  {score.sharpness:.1f}/100")
print(f"Exposure:   {score.exposure:.1f}/100")
print(f"Overall:    {score.composite:.1f}/100")
print(f"Tier:       {score.tier.upper()}")

# Output:
# Sharpness:  78.5/100
# Exposure:   82.3/100
# Overall:    80.0/100
# Tier:       HIGH
```

### Batch Processing (Phase 2)

```python
from photo_quality_analyzer import BatchProcessor

# Initialize batch processor with 4 parallel workers
processor = BatchProcessor(num_workers=4)

# Process batch with progress tracking
def show_progress(analyzed, total, failed):
    pct = (analyzed / total) * 100
    print(f"Progress: {pct:.1f}% ({analyzed}/{total}, {failed} failed)")

result = processor.process_batch(
    photo_paths=['photo1.jpg', 'photo2.jpg', 'photo3.jpg'],
    progress_callback=show_progress
)

print(f"Success: {result.successful}/{result.total_photos}")
print(f"Success rate: {result.success_rate:.1f}%")

# Access scores
for path, score in result.scores:
    print(f"{path}: {score.composite:.1f}")
```

### Library Scanning (Phase 2)

```python
from photo_quality_analyzer import LibraryScanner

# Initialize scanner
scanner = LibraryScanner()

# Scan directory recursively
photos = scanner.scan_and_collect('/Users/john/Photos')

print(f"Found {len(photos)} photos")

# Get scan statistics
stats = scanner.get_stats()
print(f"Scanned {stats.directories_scanned} directories")
print(f"Skipped {stats.skipped} files")
```

### Caching Results (Phase 2)

```python
from photo_quality_analyzer import ResultCache

# Initialize cache
cache = ResultCache('quality_scores.db')

# Check if photo needs analysis
if cache.should_analyze('photo.jpg'):
    score = analyzer.analyze_photo('photo.jpg')
    cache.set('photo.jpg', score)
else:
    score = cache.get('photo.jpg')
    print(f"Using cached score: {score.composite:.1f}")

# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

### Performance Monitoring (Phase 2)

```python
from photo_quality_analyzer import PerformanceMonitor

# Monitor analysis performance
with PerformanceMonitor() as monitor:
    for photo in photos:
        score = analyzer.analyze_photo(photo)
        monitor.record_photo()

monitor.print_summary()
# Output:
# Performance Summary:
# - Duration: 45.23s
# - Photos processed: 1000
# - Rate: 22.1 photos/sec
# - Memory: 125.3 MB -> 245.7 MB
```

### Custom Configuration

```python
from photo_quality_analyzer import create_custom_config

# Create custom weights (prioritize sharpness more)
config = create_custom_config(
    weights={'sharpness': 0.8, 'exposure': 0.2}
)

analyzer = PhotoQualityAnalyzer(config=config)
score = analyzer.analyze_photo('action_shot.jpg')
```

### Preset Configurations

```python
from photo_quality_analyzer import (
    get_conservative_config,    # Higher thresholds
    get_permissive_config,      # Lower thresholds
    get_sharpness_focused_config,
    get_exposure_focused_config
)

# Use conservative config (only highest quality)
analyzer = PhotoQualityAnalyzer(config=get_conservative_config())
```

## Architecture

```
src/photo_quality_analyzer/
├── __init__.py              # Main exports
├── analyzer.py              # Main analyzer interface
├── config.py                # Configuration and presets
├── batch_processor.py       # [Phase 2] Parallel batch processing
├── cache.py                 # [Phase 2] Result caching (SQLite)
├── scanner.py               # [Phase 2] Photo library scanning
├── performance.py           # [Phase 2] Performance monitoring
└── metrics/
    ├── __init__.py
    ├── sharpness.py         # Sharpness detection (Laplacian variance)
    ├── exposure.py          # Exposure analysis (histogram)
    └── composite.py         # Composite scoring
```

### Module Design Principles

1. **Single Responsibility:** Each module handles one metric
2. **No Cross-Dependencies:** Metrics are independent
3. **100% Unit Testable:** Pure functions with clear interfaces
4. **Type Hints:** All functions have complete type annotations
5. **Comprehensive Documentation:** Docstrings with examples

## API Reference

### PhotoQualityAnalyzer

Main analyzer class for photo quality assessment.

**Methods:**

- `analyze_photo(photo_path)` - Analyze photo from file path
- `analyze_image(image)` - Analyze numpy array image
- `analyze_batch(photo_paths)` - Analyze multiple photos
- `get_config()` - Get current configuration
- `update_config(config)` - Update configuration

**Example:**

```python
analyzer = PhotoQualityAnalyzer()
score = analyzer.analyze_photo('photo.jpg')
```

### QualityScore

Dataclass containing all quality metrics.

**Attributes:**

- `sharpness: float` - Sharpness score (0-100)
- `exposure: float` - Exposure score (0-100)
- `composite: float` - Overall quality score (0-100)
- `tier: str` - Quality tier ('high', 'acceptable', 'low')

**Methods:**

- `to_dict()` - Convert to dictionary for serialization

### Metrics Functions

**Sharpness:**

```python
from photo_quality_analyzer.metrics import calculate_sharpness_score

score = calculate_sharpness_score(image)  # Returns 0-100
```

**Exposure:**

```python
from photo_quality_analyzer.metrics import calculate_exposure_score

score = calculate_exposure_score(image)  # Returns 0-100
```

**Composite:**

```python
from photo_quality_analyzer.metrics import calculate_quality_score

score = calculate_quality_score(
    sharpness=80,
    exposure=60,
    weights={'sharpness': 0.6, 'exposure': 0.4}
)
```

## Configuration

### Default Configuration

```python
DEFAULT_WEIGHTS = {
    'sharpness': 0.6,  # 60% weight
    'exposure': 0.4    # 40% weight
}

DEFAULT_THRESHOLDS = {
    'high_quality_min': 70.0,
    'acceptable_min': 50.0,
    'low_quality_max': 49.0
}
```

### Custom Configuration

```python
config = create_custom_config(
    weights={'sharpness': 0.7, 'exposure': 0.3},
    thresholds={'high_quality_min': 75.0}
)
```

### Performance Settings

```python
config.performance.max_image_size = 1024      # Resize for speed
config.performance.default_batch_size = 500   # Photos per batch
config.performance.default_num_workers = 4    # Parallel workers
```

## Testing

### Run All Tests

```bash
# Run all tests
pytest tests/photo_quality_analyzer/

# With coverage report
pytest --cov=src/photo_quality_analyzer tests/photo_quality_analyzer/

# Run specific test file
pytest tests/photo_quality_analyzer/test_sharpness.py
```

### Test Structure

```
tests/photo_quality_analyzer/
├── test_sharpness.py        # Sharpness module tests
├── test_exposure.py         # Exposure module tests
├── test_composite.py        # Composite scoring tests
├── test_config.py           # Configuration tests
└── test_analyzer.py         # Main analyzer tests
```

**Coverage Target:** 90%+

## Algorithm Details

### Sharpness Detection

**Algorithm:** Laplacian Variance Method

- Converts image to grayscale
- Applies Laplacian operator for edge detection
- Calculates variance (higher = more edges = sharper)
- Normalizes to 0-100 scale

**Thresholds:**
- 0-30: Very blurry
- 30-50: Slightly blurry
- 50-70: Adequate
- 70-100: Sharp

### Exposure Analysis

**Algorithm:** Histogram Distribution Analysis

- Calculates pixel intensity histogram
- Detects clipped highlights (overexposure)
- Detects crushed shadows (underexposure)
- Rewards good mid-tone distribution

**Thresholds:**
- 0-30: Severely over/underexposed
- 30-50: Poor exposure
- 50-70: Acceptable
- 70-100: Well-exposed

### Composite Scoring

**Formula:**
```
Composite = (Sharpness × 0.6) + (Exposure × 0.4)
```

**Rationale:**
- Sharpness weighted higher (users care more about blur)
- Simple weighted average (no ML required)
- Easily tunable based on feedback

## Performance

**Target:** 10,000 photos in <5 minutes (33 photos/second minimum)

**Actual Performance** (MacBook Pro M1, 8-core, 16GB RAM):
- Single photo: ~22ms
- Throughput: ~45 photos/second single-threaded
- Batch (4 workers): ~180 photos/second
- 10,000 photos: ~90 seconds

**Optimizations:**
- Resize images to 1024px max (10x faster, <2% accuracy loss)
- Grayscale conversion for analysis
- Parallel batch processing
- Efficient memory management

## Command Line Usage

```bash
# Analyze single photo
python -m src.photo_quality_analyzer.analyzer photo.jpg

# Output:
# Analyzing: photo.jpg
#
# Results:
#   Sharpness:  78.5/100
#   Exposure:   82.3/100
#   Composite:  80.0/100
#   Tier:       HIGH
#
# Recommendation:
#   ✓ High quality - prioritize for curation
```

## Phase 2 Examples

### Example 1: Production Batch Analysis

```python
from photo_quality_analyzer import (
    LibraryScanner,
    BatchProcessor,
    ResultCache,
    PerformanceMonitor
)

# Step 1: Scan photo library
scanner = LibraryScanner()
photos = scanner.scan_and_collect('/Users/john/Photos')
print(f"Found {len(photos)} photos")

# Step 2: Check cache to avoid re-analyzing
cache = ResultCache('scores.db')
photos_to_analyze = [p for p in photos if cache.should_analyze(p)]
print(f"Need to analyze {len(photos_to_analyze)} photos")

# Step 3: Batch process with monitoring
processor = BatchProcessor(num_workers=4)

with PerformanceMonitor('batch_analysis') as monitor:
    result = processor.process_batch(photos_to_analyze)
    monitor.record_photo(result.successful)

# Step 4: Cache results
for path, score in result.scores:
    cache.set(path, score)

# Step 5: Print summary
monitor.print_summary()
print(f"\nAnalysis complete:")
print(f"- Processed: {result.successful}/{result.total_photos}")
print(f"- Failed: {result.failed}")
print(f"- Cache hit rate: {cache.get_stats()['hit_rate']:.1%}")
```

### Example 2: Large Library with Chunking

```python
from photo_quality_analyzer import BatchProcessor, LibraryScanner

# Scan large library
scanner = LibraryScanner()
all_photos = scanner.scan_and_collect('/Volumes/Photos')
print(f"Found {len(all_photos):,} photos")

# Process in chunks to manage memory
processor = BatchProcessor(chunk_size=1000, num_workers=4)

def show_progress(analyzed, total, failed):
    pct = (analyzed / total) * 100
    print(f"Progress: {analyzed:,}/{total:,} ({pct:.1f}%)")

result = processor.process_batch_chunked(
    all_photos,
    progress_callback=show_progress
)

print(f"\nCompleted: {result.successful:,}/{result.total_photos:,}")
print(f"Success rate: {result.success_rate:.1f}%")
```

### Example 3: Incremental Updates

```python
from photo_quality_analyzer import ResultCache, PhotoQualityAnalyzer
import os

cache = ResultCache('scores.db')
analyzer = PhotoQualityAnalyzer()

# Get recently modified photos
import time
one_week_ago = time.time() - (7 * 24 * 60 * 60)

recent_photos = [
    photo for photo in scanner.scan('/Users/john/Photos')
    if os.path.getmtime(photo) > one_week_ago
]

print(f"Found {len(recent_photos)} recently modified photos")

# Analyze only new or modified photos
for photo in recent_photos:
    if cache.should_analyze(photo):
        score = analyzer.analyze_photo(photo)
        cache.set(photo, score)
        print(f"Analyzed: {photo} -> {score.composite:.1f}")
    else:
        print(f"Cached: {photo}")
```

### Example 4: Export/Import Cache

```python
from photo_quality_analyzer import ResultCache

# Export cache to JSON
cache = ResultCache('scores.db')
cache.export_to_json('backup.json')
print(f"Exported {cache.get_stats()['total_entries']} scores")

# Import to new cache
new_cache = ResultCache('new_scores.db')
count = new_cache.import_from_json('backup.json')
print(f"Imported {count} scores")
```

## Classic Examples (Phase 1)

### Example 1: Filter Photos by Quality

```python
from pathlib import Path
from photo_quality_analyzer import PhotoQualityAnalyzer

analyzer = PhotoQualityAnalyzer()

# Get all photos in directory
photo_dir = Path('photos/')
photo_paths = list(photo_dir.glob('*.jpg'))

# Analyze all photos
scores = analyzer.analyze_batch(photo_paths)

# Separate by quality tier
high_quality = []
acceptable = []
low_quality = []

for path, score in zip(photo_paths, scores):
    if score is None:
        continue

    if score.tier == 'high':
        high_quality.append(path)
    elif score.tier == 'acceptable':
        acceptable.append(path)
    else:
        low_quality.append(path)

print(f"High quality: {len(high_quality)}")
print(f"Acceptable: {len(acceptable)}")
print(f"Low quality: {len(low_quality)}")
```

### Example 2: Compare Before/After Edits

```python
from photo_quality_analyzer import PhotoQualityAnalyzer
from photo_quality_analyzer.metrics import compare_scores

analyzer = PhotoQualityAnalyzer()

# Analyze original and edited versions
original_score = analyzer.analyze_photo('original.jpg')
edited_score = analyzer.analyze_photo('edited.jpg')

# Compare scores
comparison = compare_scores(original_score, edited_score)

print(f"Composite improvement: {-comparison['composite_diff']:.1f} points")
print(f"Sharpness change: {-comparison['sharpness_diff']:.1f}")
print(f"Exposure change: {-comparison['exposure_diff']:.1f}")

if comparison['tier_change']:
    print(f"Tier change: {comparison['tier_change']}")
```

### Example 3: Custom Scoring for Action Photos

```python
from photo_quality_analyzer import (
    PhotoQualityAnalyzer,
    create_custom_config
)

# Create config that's more forgiving of motion blur
config = create_custom_config(
    weights={'sharpness': 0.5, 'exposure': 0.5},
    thresholds={'acceptable_min': 40.0}
)

analyzer = PhotoQualityAnalyzer(config=config)

# Analyze action photos
action_photos = ['running.jpg', 'jumping.jpg', 'sports.jpg']
scores = analyzer.analyze_batch(action_photos)

# Even slightly blurry action shots may be acceptable
acceptable_count = sum(1 for s in scores if s and s.tier != 'low')
print(f"Acceptable action shots: {acceptable_count}/{len(action_photos)}")
```

## Phase 2 Implementation Summary

### Completed Components

1. **BatchProcessor** (100% test coverage)
   - Parallel processing with ProcessPoolExecutor
   - Progress tracking with callbacks
   - Error handling (individual failures don't stop batch)
   - Chunked processing for memory efficiency
   - Path validation

2. **ResultCache** (96% test coverage)
   - SQLite backend for persistent storage
   - SHA-256 hash-based photo identification
   - TTL/expiration support
   - Hit/miss statistics
   - Import/export functionality

3. **LibraryScanner** (86% test coverage)
   - Recursive directory scanning
   - Support for .jpg, .jpeg, .png, .heic, .heif
   - File validation (size, format, readability)
   - Ignore patterns (hidden files, system folders)
   - Scan statistics

4. **PerformanceMonitor** (96% test coverage)
   - Context manager pattern
   - Processing time tracking
   - Memory usage monitoring
   - Photos per second calculation
   - Aggregate statistics

### Test Results

```
Phase 2 Test Coverage:
- BatchProcessor:      105 statements, 100% coverage
- ResultCache:         151 statements,  96% coverage
- LibraryScanner:      168 statements,  86% coverage
- PerformanceMonitor:  120 statements,  96% coverage

Total: 134 tests passing
Overall project coverage: 91%
```

## Future Enhancements (Phase 3+)

- **Composition scoring:** Rule of thirds, face detection
- **Machine learning model:** Learn from human ratings
- **User feedback loop:** Improve algorithm based on corrections
- **Specialized scoring:** Different rules for portraits/landscapes/action
- **Screenshot detection:** Filter out non-photos
- **Night photo handling:** Special scoring for low-light images

## Troubleshooting

### Import Errors

```bash
# Make sure src directory is in Python path
export PYTHONPATH="${PYTHONPATH}:/Users/chrispark/dev/projects/remember-twelve"

# Or install as package (development mode)
pip install -e .
```

### Low Scores for Good Photos

- Check if image is being resized too aggressively
- Try adjusting thresholds with `create_custom_config()`
- Review edge case handling in configuration

### Slow Performance

- Ensure images are being resized (check `max_image_size` config)
- Use batch processing for multiple photos
- Consider increasing `num_workers` for parallel processing

## Contributing

This implementation follows modular design principles:

1. **Single responsibility per module**
2. **Clear interfaces with type hints**
3. **No cross-dependencies between metrics**
4. **100% unit testable**
5. **Comprehensive documentation**

When adding features:
- Write tests first (TDD approach)
- Add type hints to all functions
- Include docstrings with examples
- Update this README

## License

Part of the Remember Twelve project.

## Contact

For questions or issues, refer to the design document:
`/Users/chrispark/dev/projects/remember-twelve/4-technology/photo-quality-analyzer-design.md`
