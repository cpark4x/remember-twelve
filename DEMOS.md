# Photo Quality Analyzer - Demo Scripts

This directory contains demo scripts to showcase the Remember Twelve Photo Quality Analyzer (Phase 1 + Phase 2).

## Quick Start

### Analyze a Single Photo
```bash
python demo_quality_analyzer.py ~/Photos/vacation.jpg --detailed
```

### Analyze a Directory
```bash
python demo_quality_analyzer.py ~/Photos --limit 100
```

## Demo Scripts

### 1. `demo_quality_analyzer.py` (✨ Recommended)

**The unified, production-ready demo** that showcases both Phase 1 (Core Algorithm) and Phase 2 (Infrastructure).

#### Features:
- ✅ Single photo or directory analysis
- ✅ Parallel batch processing (4 workers)
- ✅ Intelligent caching (98% hit rate!)
- ✅ Visual quality distribution
- ✅ Performance monitoring
- ✅ Export to JSON
- ✅ Beautiful progress bars and charts

#### Usage:

```bash
# Analyze a single photo with detailed breakdown
python demo_quality_analyzer.py photo.jpg --detailed

# Analyze a directory (first 100 photos)
python demo_quality_analyzer.py ~/Photos

# Analyze 500 photos
python demo_quality_analyzer.py ~/Photos --limit 500

# Force re-analysis (ignore cache)
python demo_quality_analyzer.py ~/Photos --no-cache

# Export results to JSON
python demo_quality_analyzer.py ~/Photos --export results.json

# Show detailed per-photo scores (for small batches)
python demo_quality_analyzer.py ~/Photos --limit 20 --detailed
```

#### Example Output:

**Single Photo:**
```
======================================================================
  📸 Photo Quality Analysis
======================================================================

Photo: family_reunion.jpg

Quality Scores
──────────────────────────────────────────────────────────────────────
  Sharpness:  ✅ [🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜]  78.5
  Exposure:   ✅ [🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜]  92.3
  Composite:  ✅ [🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜⬜⬜⬜]  84.0

  Overall Tier: HIGH

Recommendation: ✅ Excellent quality! Perfect for Twelve curation.
```

**Directory:**
```
======================================================================
  📸 Photo Library Analysis
======================================================================

🔍 Scanning library...
   Found: 150 photos

💾 Cache check...
   Cached: 147 photos (98% hit rate!)
   To analyze: 3 photos

⚡ Analyzing photos...
   [████████████████████████████████████████] 100.0% 3/3

Quality Distribution
──────────────────────────────────────────────────────────────────────
   Total: 150 photos
   ✅ High quality (70-100):      89  ( 59.3%)
   ⚠️  Acceptable (50-69):        48  ( 32.0%)
   ❌ Low quality (0-49):         13  (  8.7%)

   Visual Distribution:
   🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟥🟥🟥🟥

💡 Recommendation: 89 photos ready for Twelve curation
```

---

### 2. `test_analyzer_demo.py`

**Simple Phase 1 demo** - Basic quality analysis without infrastructure.

#### Usage:
```bash
python test_analyzer_demo.py photo1.jpg photo2.jpg photo3.jpg
```

#### Output:
```
Analyzing: family_photo.jpg
  Sharpness: 85.2/100
  Exposure: 78.9/100
  Composite: 82.7/100
  Tier: ✅ HIGH
```

---

### 3. `test_phase2_demo.py`

**Phase 2 infrastructure demo** - Shows full production pipeline.

#### Usage:
```bash
python test_phase2_demo.py ~/Photos 100
```

#### Features:
- Library scanner
- Batch processor with progress tracking
- Result caching
- Performance monitoring
- Detailed statistics

---

## Understanding the Scores

### Quality Metrics

**Sharpness (0-100)** - 60% weight
- Measures focus and clarity using Laplacian variance
- High = Sharp, crisp details
- Low = Blurry, out of focus

**Exposure (0-100)** - 40% weight
- Measures lighting balance using histogram analysis
- High = Well-exposed, good contrast
- Low = Too dark, too bright, or washed out

**Composite Score** = (Sharpness × 0.6) + (Exposure × 0.4)

### Quality Tiers

| Tier | Score Range | Symbol | Recommendation |
|------|-------------|--------|----------------|
| **High** | 70-100 | ✅ | Prioritize for Twelve curation |
| **Acceptable** | 50-69 | ⚠️ | Include if needed for diversity |
| **Low** | 0-49 | ❌ | Exclude from curation |

---

## Performance

**Phase 1 (Algorithm):**
- Single photo: ~22ms
- Pure algorithm: ~180 photos/sec

**Phase 2 (Infrastructure):**
- Parallel processing: 4 workers
- Throughput: 68-72 photos/sec (real-world)
- 10,000 photos: ~2-3 minutes
- Cache hit rate: 98%+ on subsequent runs

**Memory:**
- Single photo: <100MB
- 100 photos batch: <150MB
- 1,000 photos batch: <500MB (with chunking)

---

## Cache Behavior

The demos use an SQLite cache to avoid re-analyzing photos:

**Cache Location:**
- Demo scripts: `/tmp/remember_twelve_cache.db`
- Phase 2 demo: `/tmp/remember_twelve_demo.db`

**Cache Invalidation:**
- Photos are identified by SHA-256 hash
- If you edit a photo, it will be re-analyzed automatically
- Use `--no-cache` flag to force re-analysis

**Clear Cache:**
```bash
rm /tmp/remember_twelve_cache.db
```

---

## Export Format

The `--export` option creates a JSON file:

```json
{
  "total": 150,
  "summary": {
    "high": 89,
    "acceptable": 48,
    "low": 13
  },
  "photos": [
    {
      "name": "family_reunion.jpg",
      "composite": 84.0,
      "sharpness": 78.5,
      "exposure": 92.3,
      "tier": "high"
    },
    ...
  ]
}
```

---

## Tips

### Best Practices

1. **Start small**: Test with `--limit 20` first to see how it works
2. **Use cache**: Let the cache do its magic on large libraries
3. **Export results**: Save JSON for later analysis or integration
4. **Monitor progress**: Watch the progress bar and performance metrics

### Common Use Cases

**Quick quality check:**
```bash
python demo_quality_analyzer.py ~/Desktop
```

**Curate yearly photos:**
```bash
python demo_quality_analyzer.py ~/Photos/2024 --limit 1000 --export 2024_scores.json
```

**Find your best photos:**
```bash
python demo_quality_analyzer.py ~/Photos --detailed | grep "🏆"
```

**Performance testing:**
```bash
python demo_quality_analyzer.py ~/Photos --limit 1000 --no-cache
```

---

## Troubleshooting

**"No photos found"**
- Check that the directory contains .jpg, .jpeg, .png, .heic, or .heif files
- Hidden files (starting with `.`) are automatically skipped

**"Failed to analyze photo"**
- Some image files may be corrupted or in unsupported formats
- The demo continues processing other photos

**Slow performance**
- First run is always slower (no cache)
- Large images (>10MB) take longer to process
- Reduce `--limit` for faster testing

**High memory usage**
- Normal for large batches (photos loaded in parallel)
- Use smaller `--limit` values if memory is constrained

---

## What's Next?

These demos showcase **Feature 1.1: Photo Quality Analyzer** from Remember Twelve.

**Coming Soon:**
- Feature 1.2: Emotional Significance Detection
- Feature 1.3: Composition Analysis
- Curation Engine (automatically select best 12 photos/year)
- Multi-circle organization
- Reflection interface

---

## Learn More

- [Photo Quality Analyzer README](src/photo_quality_analyzer/README.md)
- [Architecture Documentation](4-technology/photo-quality-analyzer-design.md)
- [Feature Specification](2-product/Features/epic-1-ai-curation/feature-1.1-photo-quality-analyzer.md)
- [Remember Twelve Vision](1-vision/Vision.md)

---

**Remember Twelve**: Preserve your year in twelve unforgettable moments.

*Built with spec-driven development using [amplifier](amplifier/) patterns.*
