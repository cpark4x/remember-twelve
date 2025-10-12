# User Story 1.2.1: Detect Faces and Emotions in Photos

> Part of [Feature 1.2: Emotional Significance Detector](../../../Features/epic-1-ai-curation/feature-1.2-emotional-significance-detector.md)

---

### User Story

**As a** Remember Twelve user
**I want** the system to detect faces and positive emotions in my photos
**So that** photos with meaningful human moments are prioritized for my Twelve curation

---

### Acceptance Criteria

- [ ] Given a photo with 1+ clear faces, when analyzed, then all visible faces are detected with bounding boxes
- [ ] Given a photo with smiling faces, when analyzed, then positive emotions are identified for each face
- [ ] Given a photo with no faces, when analyzed, then the system returns emotional significance score of 0
- [ ] Given a photo with multiple people, when analyzed, then face count is accurately reported (1-20+)
- [ ] Given a photo with embracing people, when analyzed, then intimacy/closeness is detected
- [ ] Given a batch of 100 photos, when analyzed, then processing completes in <2 seconds (50+ photos/sec)
- [ ] Given previously analyzed photos, when re-analyzed, then cached results are returned instantly

---

### Technical Notes

**Implementation Approach:**
- Use OpenCV for face detection (Haar Cascades or DNN-based)
- Use facial landmark analysis or pre-trained model for smile detection
- Calculate face coverage (% of photo occupied by faces)
- Measure face proximity for intimacy detection
- Return structured EmotionalScore dataclass

**Key Components:**
```python
@dataclass
class EmotionalScore:
    face_count: int              # 0-20+
    face_coverage: float         # 0.0-1.0 (% of image)
    emotion_score: float         # 0-100 (positive emotion strength)
    intimacy_score: float        # 0-100 (closeness/affection)
    engagement_score: float      # 0-100 (facing camera)
    composite: float             # 0-100 (weighted total)
    tier: str                    # 'high', 'medium', 'low'
    metadata: dict               # Face positions, confidences, etc.
```

**Detection Pipeline:**
1. Load image (reuse from quality analyzer)
2. Detect faces using OpenCV
3. For each face: extract region, analyze emotion
4. Calculate proximity between faces
5. Calculate engagement (facing camera)
6. Compute composite emotional significance score

---

### Design Reference

**Scoring Formula:**
```
Face Presence (0-30):
  - 0 faces: 0
  - 1 face: 20
  - 2 faces: 25
  - 3+ faces: 30

Positive Emotion (0-40):
  - No smiles: 0
  - Subtle: 20
  - Clear: 30
  - Genuine joy: 40

Intimacy (0-20):
  - Distant: 0
  - Moderate: 10
  - Close: 15
  - Embracing: 20

Engagement (0-10):
  - Looking away: 0
  - Some eye contact: 5
  - Facing camera: 10

Composite = sum(components)
```

**Tier Classification:**
- High: 70-100 (strong emotional significance)
- Medium: 40-69 (moderate emotional significance)
- Low: 0-39 (minimal emotional significance)

---

### Dependencies

- **Depends on:** Feature 1.1 (Photo Quality Analyzer) - ✅ Complete
- **Blocks:** Story 1.2.2 (Score Emotional Significance) - needs detection results
- **Related:** Batch processor, cache, scanner (reuse infrastructure)

---

### Testing Notes

**Test Scenarios:**

1. **Single person, smiling**
   - Input: Selfie with clear smile
   - Expected: 1 face, high emotion score, 60+ composite

2. **Couple embracing**
   - Input: Two people in close physical contact
   - Expected: 2 faces, high intimacy score, 80+ composite

3. **Large group, all smiling**
   - Input: Birthday party with 5+ people
   - Expected: 5+ faces, high emotion, 90+ composite

4. **Landscape, no people**
   - Input: Mountain scenery
   - Expected: 0 faces, 0 composite score

5. **Person from behind (no face visible)**
   - Input: Person facing away
   - Expected: 0 faces detected, 0 score

6. **Serious portrait (no smile)**
   - Input: Professional headshot, neutral expression
   - Expected: 1 face, low emotion, 20-30 composite

**Edge Cases:**

- Very small faces (<5% of photo) - should be ignored
- Partially occluded faces (profile, sunglasses) - best effort detection
- Face-like objects (statues, masks, artwork) - acceptable false positives in MVP
- Poor lighting / blurry faces - graceful degradation
- Multiple faces at different depths - all should be detected
- Babies/children vs. adults - should work equally well

**Performance Tests:**
- 100 photos with faces: <2 seconds
- 1000 photos mixed: <20 seconds
- Single photo: <50ms

**Accuracy Tests:**
- Face detection: >90% on clear faces
- Smile detection: >80% on clear smiles
- False positives: <10% on non-face photos

---

### Estimated Effort

**Phase 1 (Core Detection):** 2-3 days
- Face detection implementation
- Emotion recognition
- Scoring algorithm
- Unit tests

**Phase 2 (Integration):** 1 day
- Batch processing integration
- Cache integration
- Demo updates

**Total:** 3-4 days

---

### Example Output

```python
# Analyze a birthday party photo
from emotional_significance import EmotionalAnalyzer

analyzer = EmotionalAnalyzer()
score = analyzer.analyze_photo('birthday_party.jpg')

print(f"Faces detected: {score.face_count}")
# Output: Faces detected: 5

print(f"Emotion score: {score.emotion_score}/100")
# Output: Emotion score: 95/100 (all smiling)

print(f"Intimacy: {score.intimacy_score}/100")
# Output: Intimacy: 80/100 (close together)

print(f"Composite: {score.composite}/100")
# Output: Composite: 95/100

print(f"Tier: {score.tier}")
# Output: Tier: HIGH

print(f"Recommendation: This photo has strong emotional significance!")
```

---

### Metadata & Change History

| Version | Date       | Author | Changes                                       |
| ------- | ---------- | ------ | --------------------------------------------- |
| v1.0    | 2025-10-11 | Claude | Initial user story for face/emotion detection |

