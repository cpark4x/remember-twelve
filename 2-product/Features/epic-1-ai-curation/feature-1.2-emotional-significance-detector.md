# Feature 1.2: Emotional Significance Detector

### Epic Context

**Parent Epic:** [Epic 1: AI-Powered Photo Curation Engine](../../Epics/epic-1-ai-powered-photo-curation-engine.md)
**Epic Objective:** Build an AI system that automatically identifies the best photos for curation based on technical quality, emotional significance, and composition

---

### Feature Overview

**What:** An AI-powered detector that analyzes photos for emotional significance by identifying faces, emotions (smiles, joy), people density, and intimate moments (embraces, closeness).

**Why:** Technical quality alone isn't enough. A perfectly sharp, well-exposed photo of an empty room isn't memorable. Photos with smiling faces, embraces, and emotional connections are what users want to preserve in their Twelve. This feature ensures the curation engine prioritizes meaningful human moments.

**Success Criteria:**
- Accurately detects faces in photos (>90% precision on clear faces)
- Identifies positive emotions (smiles, happiness) with >80% accuracy
- Calculates emotional significance scores (0-100) combining multiple signals
- Processes photos at similar speed to quality analyzer (~50+ photos/sec)
- Integrates seamlessly with existing QualityScore system

---

### User Stories

Link to related user stories:

- [Story 1.2.1](../../UserStories/epic-1-ai-curation/feature-1.2/us-1.2.1-detect-faces-and-emotions.md) - Detect faces and identify positive emotions
- [Story 1.2.2](../../UserStories/epic-1-ai-curation/feature-1.2/us-1.2.2-score-emotional-significance.md) - Calculate emotional significance score
- [Story 1.2.3](../../UserStories/epic-1-ai-curation/feature-1.2/us-1.2.3-combine-with-quality-scores.md) - Integrate with quality analyzer for unified scoring

---

### Technical Requirements

#### Core Detection Capabilities

1. **Face Detection**
   - Detect faces in photos (position, size, count)
   - Track face coverage (% of photo occupied by faces)
   - Support multiple faces (1-20+ people)
   - Handle various lighting and angles

2. **Emotion Recognition**
   - Identify positive emotions (smiles, happiness, joy)
   - Detect engagement (eye contact, facing camera)
   - Score emotion intensity (subtle smile vs. big grin)

3. **Intimacy Detection**
   - Measure people proximity (embraces, closeness)
   - Detect physical affection indicators
   - Identify group cohesion

4. **People Density Analysis**
   - Count people in photo
   - Classify: solo (1), couple (2), small group (3-5), large group (6+)
   - Weight significance by group size

#### Scoring Algorithm

**Emotional Significance Score (0-100):**
```
Components:
- Face Presence (0-30 points):
  - No faces: 0
  - 1 face: 20
  - 2 faces: 25
  - 3+ faces: 30

- Positive Emotion (0-40 points):
  - No smiles: 0
  - Subtle smiles: 20
  - Clear smiles: 30
  - Genuine joy: 40

- Intimacy/Closeness (0-20 points):
  - Distant: 0
  - Moderate proximity: 10
  - Close/touching: 15
  - Embracing: 20

- Engagement (0-10 points):
  - Looking away: 0
  - Some eye contact: 5
  - All facing camera: 10

Composite = Face Presence + Emotion + Intimacy + Engagement
```

#### Technical Stack

- **Face Detection:** OpenCV Haar Cascades or DNN-based detector (fast, local)
- **Emotion Recognition:** Pre-trained model or facial landmark analysis
- **Alternative:** MediaPipe Face Detection (Google, optimized for speed)
- **Image Processing:** OpenCV, NumPy (already in stack)
- **No cloud APIs:** All processing local for privacy

#### Performance Targets

- Single photo: <50ms
- Batch processing: 50+ photos/sec (parallel)
- Memory: <200MB for 100 photos
- No internet required (fully local)

#### Privacy & Ethics

- All processing happens locally (no data sent to cloud)
- No face identification/recognition (only detection and emotion)
- No biometric data stored (only scores)
- Optional feature (users can disable)

---

### Dependencies

- **Blocked by:**
  - Feature 1.1 (Photo Quality Analyzer) - ✅ Complete

- **Blocks:**
  - Feature 2.1 (Multi-Circle Photo Organization) - needs combined scores
  - Curation Engine - needs emotional + quality scores for ranking

- **Related:**
  - Feature 1.3 (Composition Analysis) - will also contribute to final ranking
  - Infrastructure Phase 2 - reuse BatchProcessor, Cache, Scanner

---

### Success Metrics

**Accuracy Metrics:**
- Face detection precision: >90% on photos with clear faces
- Emotion detection accuracy: >80% on smiling faces
- False positive rate: <10% (no faces detected in non-people photos)

**Performance Metrics:**
- Processing speed: 50+ photos/sec
- Single photo latency: <50ms
- Memory usage: <200MB per 100 photos

**User Impact:**
- % of curated photos with faces: Target 70%+ (vs. random ~40%)
- % of curated photos with smiles: Target 60%+ (vs. random ~20%)
- User satisfaction: "The Twelve captured meaningful moments" >80% agree

**Technical Metrics:**
- Test coverage: >90%
- Integration with quality analyzer: seamless
- Cache reuse: emotional scores cached like quality scores

---

### Implementation Phases

#### Phase 1: Core Detection (MVP)
- Face detection using OpenCV
- Smile detection (basic emotion)
- Face count and coverage
- Emotional significance score calculation
- Unit tests

#### Phase 2: Infrastructure Integration
- Integrate with BatchProcessor
- Add to ResultCache
- Update demo scripts
- Performance optimization

#### Phase 3: Advanced Emotions (Future)
- Multiple emotion types (joy, surprise, love)
- Emotion intensity levels
- Group interaction detection
- Machine learning model integration

---

### Definition of Done

- [x] Feature specification complete
- [ ] User stories created
- [ ] Architecture designed (zen-architect)
- [ ] Phase 1 implementation complete (modular-builder)
- [ ] Phase 2 integration complete
- [ ] Unit tests written (>90% coverage)
- [ ] Integration tests passing
- [ ] Performance benchmarks met (50+ photos/sec)
- [ ] Demo updated with emotional scores
- [ ] Documentation complete
- [ ] Tested on real photos with faces

---

### Design Decisions

**Why OpenCV over cloud APIs?**
- Privacy: All processing local
- Cost: No API fees
- Speed: No network latency
- Offline: Works without internet
- Control: Full control over algorithm

**Why not deep learning models?**
- Phase 1 uses OpenCV (lightweight, proven)
- Phase 3 can add ML models if needed
- Start simple, add complexity only if necessary

**Why 0-100 scoring?**
- Consistent with quality analyzer
- Easy to understand and compare
- Allows weighted combination with quality scores

**Weighting rationale:**
- Emotion (40%) = Most important signal of significance
- Face presence (30%) = Foundation (no faces = not significant)
- Intimacy (20%) = Strong indicator of meaningful moments
- Engagement (10%) = Nice-to-have bonus

---

### Open Questions

1. **Minimum face size:** Should we ignore very small faces (<5% of photo)?
   - **Decision:** Yes, likely background people, not primary subjects

2. **Group size weighting:** Should large groups (10+ people) score higher?
   - **Decision:** Cap at 3+ faces for now, revisit in Phase 3

3. **False positives:** How to handle face-like objects (statues, masks)?
   - **Decision:** Accept some false positives in MVP, refine later

4. **Multiple people, no smiles:** How to score serious family portraits?
   - **Decision:** Face presence (30 pts) ensures they score moderately well

---

### Example Scenarios

**Scenario 1: Birthday party (5 people, all smiling)**
- Face Presence: 30 (3+ faces)
- Emotion: 40 (clear smiles)
- Intimacy: 15 (close together)
- Engagement: 10 (facing camera)
- **Total: 95/100** ✅ High emotional significance

**Scenario 2: Landscape (no people)**
- Face Presence: 0
- Emotion: 0
- Intimacy: 0
- Engagement: 0
- **Total: 0/100** ❌ No emotional significance (rely on quality only)

**Scenario 3: Solo selfie (1 person, smiling)**
- Face Presence: 20 (1 face)
- Emotion: 30 (clear smile)
- Intimacy: 0 (solo)
- Engagement: 10 (facing camera)
- **Total: 60/100** ⚠️ Acceptable emotional significance

**Scenario 4: Couple embracing (2 people, smiling)**
- Face Presence: 25 (2 faces)
- Emotion: 40 (genuine joy)
- Intimacy: 20 (embracing)
- Engagement: 5 (some looking away)
- **Total: 90/100** ✅ High emotional significance

---

### Metadata & Change History

| Version | Date       | Author | Changes                                                    |
| ------- | ---------- | ------ | ---------------------------------------------------------- |
| v1.0    | 2025-10-11 | Claude | Initial feature spec for Emotional Significance Detection |

