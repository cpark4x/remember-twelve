# User Story 1.1.1: Score Photo Library on First Import

> Part of [Feature 1.1: Photo Quality Analyzer](../../Features/epic-1-ai-curation/feature-1.1-photo-quality-analyzer.md)

---

### User Story

**As a** new Remember Twelve user
**I want** my photo library automatically analyzed for quality when I first import it
**So that** the AI can exclude low-quality photos from my Twelve without any manual work

---

### Acceptance Criteria

- [ ] Given I connect my photo library (Google Photos, iCloud, or local), when the import completes, then quality analysis runs automatically in the background
- [ ] Given quality analysis is running, when I check the app, then I see a progress indicator showing "Analyzing X of Y photos"
- [ ] Given analysis completes, when I view my library, then low-quality photos (<50 score) are flagged and excluded from curation by default
- [ ] Given analysis fails for a photo (corrupted file, unsupported format), when I view my library, then that photo is skipped with a warning logged (but import continues)
- [ ] Given I have 10,000 photos, when analysis runs, then it completes in under 10 minutes

---

### Technical Notes

**Implementation Approach**:
- Trigger quality analysis as background job after photo import
- Use Celery/background worker to avoid blocking UI
- Process in batches of 500 photos to manage memory
- Store results in `photo_quality_scores` table

**Libraries**:
- OpenCV for sharpness detection
- Pillow for exposure/histogram analysis
- NumPy for efficient array operations

**Performance**:
- Target 30-50 photos/second
- Use thumbnail analysis (1024px max) to speed up processing
- Parallelize across CPU cores

---

### Design Reference

**UI Components**:
- Progress bar during analysis: "Analyzing photos... 2,347 of 10,000"
- Optional: "Quality analysis complete! We analyzed 10,000 photos and found 9,234 high-quality images for curation."

*Design mockups to be added in Figma*

---

### Dependencies

- **Depends on**: Photo import infrastructure (must exist to trigger analysis)
- **Depends on**: Database schema for `photo_quality_scores` table
- **Blocks**: Feature 1.4 (Twelve Selection Algorithm) - needs quality scores
- **Related**: Feature 1.2 (Emotional Significance Detector) - runs after quality analysis

---

### Testing Notes

**Test Scenarios**:

1. **Happy Path**: Import 100 photos, verify all are analyzed and scored
2. **Large Library**: Import 10,000 photos, verify completes in <10 minutes
3. **Mixed Quality**: Import intentionally blurry/dark photos, verify low scores
4. **Interrupted Analysis**: Stop analysis mid-run, restart, verify resumes correctly
5. **Corrupted Photos**: Include 1-2 corrupted images, verify they're skipped gracefully

**Edge Cases**:
- Photos with no EXIF data
- Non-standard formats (HEIC, WebP, RAW)
- Extremely large photos (50MB+)
- Screenshots and text-heavy images

---

### Estimated Effort

**3-5 days** (1 sprint)

**Breakdown**:
- Day 1: Set up OpenCV, implement sharpness detection
- Day 2: Implement exposure analysis
- Day 3: Integrate with photo import pipeline, background job
- Day 4: Testing and optimization
- Day 5: Bug fixes, edge case handling

---

### Metadata & Change History

| Version | Date       | Author     | Changes                     |
| ------- | ---------- | ---------- | --------------------------- |
| v1.0    | 2025-10-10 | Chris Park | Initial user story created. |
