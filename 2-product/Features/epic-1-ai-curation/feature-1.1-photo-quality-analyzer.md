# Feature 1.1: Photo Quality Analyzer

### Epic Context

**Parent Epic:** [Epic 1: AI-Powered Photo Curation Engine](../../Epics/epic-1-ai-curation-engine.md)

**Epic Objective:** Build an AI-powered curation engine that automatically selects twelve meaningful photos per year for each circle using quality, emotion, diversity, and significance signals.

---

### Feature Overview

**What:** An automated photo quality analysis system that scores photos based on technical quality metrics (sharpness, exposure, composition) to filter out poor-quality images from curation consideration.

**Why:**
- **User Benefit**: Ensures the AI only selects high-quality photos, preventing blurry/dark/poorly composed images from appearing in their Twelve
- **Business Value**: Quality scoring is foundational to the entire curation engine; without it, AI selections feel random and untrustworthy
- **Technical Foundation**: This is the first filter in the curation pipeline; every photo runs through quality analysis before other signals (emotion, diversity)

**Success Criteria:**
- 90%+ of selected photos are rated "good quality" or better by users
- System can score 10,000 photos in <5 minutes
- Quality scores correlate with human judgment (validation on 500 photo test set)

---

### Detailed Requirements

#### Quality Metrics

The analyzer evaluates photos across three dimensions:

1. **Sharpness/Focus**
   - Detect blur using Laplacian variance or similar edge detection
   - Flag out-of-focus images (common in candid family photos)
   - Score: 0-100 (0=completely blurred, 100=tack sharp)

2. **Exposure**
   - Analyze histogram to detect over/underexposure
   - Check for clipped highlights or crushed shadows
   - Handle HDR/high-contrast scenes gracefully
   - Score: 0-100 (0=severely over/under exposed, 100=well-exposed)

3. **Composition** (Optional for MVP)
   - Rule of thirds detection
   - Face/subject centering
   - Horizon leveling
   - Score: 0-100 (0=poor composition, 100=excellent)

#### Composite Quality Score

**Formula (MVP)**:
```
Quality Score = (Sharpness × 0.6) + (Exposure × 0.4)
```

*Composition deferred to V2*

**Thresholds**:
- **High Quality**: 70-100 → Prioritize for curation
- **Acceptable**: 50-69 → Include if needed for diversity
- **Low Quality**: 0-49 → Exclude from curation entirely

#### Performance Requirements

- **Throughput**: 30-50 photos/second on average hardware
- **Batch Processing**: Handle 10,000 photos in <5 minutes
- **Memory**: <2GB RAM for processing batch
- **Storage**: Cache scores (don't recompute on every curation run)

#### Edge Cases

- **Very dark photos** (night shots): May be intentionally dark; don't penalize
- **Action shots with motion blur**: Distinguish from poor focus
- **Black & white photos**: Exposure metrics still valid
- **Screenshots/text**: May fail sharpness tests; flag separately

---

### User Stories

Link to related user stories:

- [Story 1.1.1: Score photo library on first import](../../UserStories/epic-1-ai-curation/us-1.1.1-score-library-import.md) - Initial quality analysis
- [Story 1.1.2: Display quality scores in UI](../../UserStories/epic-1-ai-curation/us-1.1.2-display-quality-scores.md) - Show scores to users (optional)
- [Story 1.1.3: Re-score edited photos](../../UserStories/epic-1-ai-curation/us-1.1.3-rescore-edited-photos.md) - Handle photo edits

---

### Technical Requirements

#### Technology Stack

**Libraries (Python)**:
- **OpenCV**: Image processing, sharpness detection
- **Pillow (PIL)**: Image loading, histogram analysis
- **NumPy**: Fast array operations

**Alternative**:
- Pre-trained models (MobileNet quality predictor) if available
- Cloud ML APIs (Google Vision, AWS Rekognition) for quality scoring

**Recommendation**: Start with OpenCV (free, fast, no API costs)

#### Data Storage

**Database Schema** (PostgreSQL):
```sql
CREATE TABLE photo_quality_scores (
  photo_id UUID PRIMARY KEY,
  sharpness_score FLOAT,  -- 0-100
  exposure_score FLOAT,   -- 0-100
  composite_score FLOAT,  -- 0-100
  analyzed_at TIMESTAMP,
  FOREIGN KEY (photo_id) REFERENCES photos(id)
);
```

**Caching**:
- Store scores in database
- Only recompute if photo metadata changes (file hash)

#### Integration Points

- **Photo Import Service**: Trigger quality analysis after upload
- **Curation Engine**: Query quality scores to filter candidates
- **Batch Processor**: Run analysis on all photos in background job

#### Performance Optimization

- **Lazy Loading**: Only analyze photos when needed (not on import)
- **Batch Processing**: Analyze in chunks of 100-500 photos
- **Parallel Processing**: Use multiprocessing for CPU-bound tasks
- **Thumbnail Analysis**: Score thumbnails (1024px) instead of full-res

---

### Dependencies

- **Blocks**:
  - Feature 1.4 (Twelve Selection Algorithm) - needs quality scores as input
  - Feature 1.6 (Curation Transparency Dashboard) - displays quality reasoning

- **Blocked by**:
  - Photo import infrastructure (must exist to have photos to analyze)
  - Photo metadata extraction (dimensions, format, EXIF data)

- **Related**:
  - Feature 1.2 (Emotional Significance Detector) - both run on same photos
  - Feature 1.3 (Temporal & Subject Diversity) - coordinates with quality filtering

---

### Open Questions

- [ ] Should we show quality scores to users? (Transparency vs. complexity)
- [ ] Do we need separate scoring for portraits vs. landscapes?
- [ ] How do we handle intentionally artistic photos (blur, overexposure)?
- [ ] Should users be able to override quality filtering? (e.g., "include this blurry photo anyway")
- [ ] What's the acceptable false positive rate? (Good photos marked as low quality)

---

### Testing Strategy

#### Unit Tests
- Test sharpness detection on synthetic blurred images
- Test exposure analysis on over/underexposed samples
- Test edge cases (black images, white images, text screenshots)

#### Validation Dataset
- **500 photos** hand-labeled by humans as "good" or "poor" quality
- Compare algorithm scores to human judgments
- Target: 85%+ agreement with human ratings

#### Performance Tests
- Benchmark 10,000 photo analysis on target hardware
- Measure memory usage under load
- Test batch processing interruption/resumption

---

### Definition of Done

- [ ] Quality analyzer library implemented with sharpness + exposure scoring
- [ ] Database schema created and migrated
- [ ] Batch processing script can analyze photo library
- [ ] Unit tests passing (90%+ coverage)
- [ ] Validation on 500 photo test set (85%+ accuracy vs. human judgment)
- [ ] Performance benchmarks met (<5 min for 10K photos)
- [ ] Code reviewed and merged to main
- [ ] Documentation: algorithm explanation, API reference
- [ ] Integration with photo import pipeline complete

---

### Metadata & Change History

| Version | Date       | Author     | Changes                                       |
| ------- | ---------- | ---------- | --------------------------------------------- |
| v1.0    | 2025-10-10 | Chris Park | Initial feature breakdown for Photo Quality Analyzer. |
