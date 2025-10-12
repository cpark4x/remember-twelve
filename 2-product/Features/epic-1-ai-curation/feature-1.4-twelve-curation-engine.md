# Feature 1.4: Twelve Curation Engine

### Epic Context

**Parent Epic:** [Epic 1: AI-Powered Photo Curation Engine](../../Epics/epic-1-ai-powered-photo-curation-engine.md)
**Epic Objective:** Build an AI system that automatically identifies the best photos for curation based on technical quality, emotional significance, and composition

---

### Feature Overview

**What:** An intelligent curation engine that automatically selects the best 12 photos from a year's worth of photos by combining quality scores, emotional significance, temporal diversity, and visual variety.

**Why:** The core promise of Remember Twelve is "preserve your year in twelve unforgettable moments." Users have thousands of photos per year - this engine makes the hard choice for them, ensuring their Twelve captures the full breadth of their year with the most meaningful, high-quality moments.

**Success Criteria:**
- Selects exactly 12 photos from a year's library
- Achieves temporal diversity (spread across months, not clustered)
- Ensures visual diversity (different scenes/subjects, not repetitive)
- Prioritizes quality + emotional significance (uses Features 1.1 + 1.2)
- 80%+ user satisfaction: "My Twelve captured my year well"
- Processes 1000+ photos in <30 seconds

---

### User Stories

Link to related user stories:

- [Story 1.4.1](../../UserStories/epic-1-ai-curation/feature-1.4/us-1.4.1-rank-photos-by-combined-score.md) - Rank photos using quality + emotional scores
- [Story 1.4.2](../../UserStories/epic-1-ai-curation/feature-1.4/us-1.4.2-ensure-temporal-diversity.md) - Spread selections across the year
- [Story 1.4.3](../../UserStories/epic-1-ai-curation/feature-1.4/us-1.4.3-ensure-visual-diversity.md) - Avoid repetitive/similar photos
- [Story 1.4.4](../../UserStories/epic-1-ai-curation/feature-1.4/us-1.4.4-generate-twelve-selections.md) - Output final curated Twelve

---

### Technical Requirements

#### Core Algorithm

**The Twelve Selection Process:**

1. **Score Combination** (Features 1.1 + 1.2)
   - Quality Score: 0-100 (sharpness + exposure)
   - Emotional Score: 0-100 (faces + emotions)
   - Combined Score: (Quality × 0.4) + (Emotional × 0.6)

2. **Temporal Grouping**
   - Group photos by month (12 buckets: Jan-Dec)
   - Goal: Select at least 1 photo per month (if available)
   - If <12 months have photos, distribute across available months

3. **Per-Month Selection**
   - Sort month's photos by combined score (highest first)
   - Apply visual diversity filter (avoid near-duplicates)
   - Select best photo(s) from each month

4. **Visual Diversity Filter**
   - Detect near-duplicate photos (similar scenes/subjects)
   - Ensure variety across the Twelve
   - Use histogram similarity or perceptual hashing

5. **Final Twelve**
   - Guarantee exactly 12 photos
   - Ranked by combined score within temporal constraints
   - Metadata: month, score, rank, diversity info

#### Curation Strategies

**Strategy 1: Balanced Distribution (Default)**
- Distribute 12 selections across 12 months (1 per month)
- If month has no photos, reallocate to other months
- Priority: Temporal spread over absolute best scores

**Strategy 2: Top-Heavy**
- Prioritize absolute best scores
- Less emphasis on temporal diversity
- Good for users with clustered photo-taking (vacations)

**Strategy 3: People-First**
- Weight emotional score higher (70% emotional, 30% quality)
- Prioritize photos with faces/people
- Good for family-focused users

**Strategy 4: Aesthetic-First**
- Weight quality score higher (70% quality, 30% emotional)
- Prioritize technical excellence
- Good for photography enthusiasts

#### Data Structures

```python
@dataclass
class PhotoCandidate:
    """A photo being considered for the Twelve."""
    photo_path: Path
    timestamp: datetime
    month: int  # 1-12
    quality_score: float  # 0-100
    emotional_score: float  # 0-100
    combined_score: float  # 0-100
    metadata: dict

@dataclass
class TwelveSelection:
    """The final curated Twelve for a year."""
    year: int
    photos: List[PhotoCandidate]  # Exactly 12
    strategy: str
    stats: dict  # Avg scores, diversity metrics
    created_at: datetime

    def to_dict(self) -> dict:
        """Export to JSON."""

    def save(self, output_path: Path):
        """Save selection to file."""
```

#### API Design

```python
class TwelveCurator:
    def __init__(self, config: Optional[CurationConfig] = None):
        """Initialize curator with config."""

    def curate_year(
        self,
        photo_library: List[Path],
        year: int,
        strategy: str = "balanced"
    ) -> TwelveSelection:
        """Curate 12 photos from a year's library."""

    def preview_candidates(
        self,
        photo_library: List[Path],
        year: int,
        top_n: int = 50
    ) -> List[PhotoCandidate]:
        """Get top N candidates before final selection."""

    def curate_with_fallback(
        self,
        photo_library: List[Path],
        year: int
    ) -> TwelveSelection:
        """Curate with automatic fallback if <12 photos."""
```

---

### Algorithm Details

#### Phase 1: Scoring and Ranking

```python
# For each photo in library
for photo in photo_library:
    # Get scores (cached if available)
    quality = quality_analyzer.analyze(photo)
    emotional = emotional_analyzer.analyze(photo)

    # Combined score (weighted)
    combined = (quality.composite * 0.4) + (emotional.composite * 0.6)

    # Create candidate
    candidate = PhotoCandidate(
        photo_path=photo,
        timestamp=get_exif_date(photo),
        month=timestamp.month,
        quality_score=quality.composite,
        emotional_score=emotional.composite,
        combined_score=combined
    )

    candidates.append(candidate)

# Sort by combined score (descending)
candidates.sort(key=lambda c: c.combined_score, reverse=True)
```

#### Phase 2: Temporal Distribution

```python
# Group by month
by_month = defaultdict(list)
for candidate in candidates:
    by_month[candidate.month].append(candidate)

# Select best from each month
monthly_selections = []
for month in range(1, 13):
    if month in by_month:
        # Get top photo from this month
        month_photos = by_month[month]
        best = max(month_photos, key=lambda c: c.combined_score)
        monthly_selections.append(best)

# If <12 months have photos, select more from active months
if len(monthly_selections) < 12:
    remaining = 12 - len(monthly_selections)
    # Get next best photos (avoiding already selected)
    ...
```

#### Phase 3: Visual Diversity

```python
def is_visually_diverse(photo_a: Path, photo_b: Path, threshold: float = 0.15) -> bool:
    """Check if two photos are visually different enough."""
    # Method 1: Histogram similarity
    hist_a = calculate_histogram(photo_a)
    hist_b = calculate_histogram(photo_b)
    similarity = compare_histograms(hist_a, hist_b)

    # Method 2: Perceptual hash (dhash)
    hash_a = dhash(photo_a)
    hash_b = dhash(photo_b)
    hamming_distance = count_differing_bits(hash_a, hash_b)

    # Diverse if sufficiently different
    return similarity < (1 - threshold) or hamming_distance > 10

# Apply diversity filter
final_twelve = []
for candidate in monthly_selections:
    if all(is_visually_diverse(candidate.photo_path, selected.photo_path)
           for selected in final_twelve):
        final_twelve.append(candidate)
```

---

### Success Metrics

**Selection Quality:**
- Average combined score: Target >70/100
- Photos with faces: Target >60% (emotional moments prioritized)
- High-quality photos: Target >80% in High tier

**Diversity Metrics:**
- Temporal diversity: Target 10+ months represented (out of 12)
- Visual diversity: No near-duplicates in Twelve
- Scene variety: Multiple different scenes/locations

**User Satisfaction:**
- "My Twelve captured my year well": >80% agree
- "I would keep most of these": >75% agree
- "Better than I would pick manually": >60% agree

**Performance:**
- 1000 photos processed: <30 seconds
- 10,000 photos processed: <5 minutes
- Caching utilized: >95% on repeated curation

---

### Edge Cases

**1. Fewer than 12 photos in year**
- Return all available photos
- Clearly indicate it's incomplete
- Suggest importing more photos

**2. Photos clustered in one month (vacation)**
- Allow multiple from same month if quality justifies
- Still maintain some temporal spread
- Use visual diversity heavily

**3. All photos are low quality**
- Still select 12 best available
- Warning: "Photos may not meet quality standards"
- Suggest retaking/re-importing

**4. No photos with faces (all landscapes)**
- Emotional score will be low (0-20)
- Quality score becomes dominant
- Still select 12 best landscapes

**5. Near-duplicate photos (burst mode)**
- Visual diversity filter removes duplicates
- Select best from burst sequence
- Based on sharpness/composition

**6. Missing EXIF dates**
- Use file modification date as fallback
- Group by quarter if month unknown
- Warning: "Some dates estimated"

---

### Curation Configuration

```python
@dataclass
class CurationConfig:
    # Scoring weights
    quality_weight: float = 0.4  # 40%
    emotional_weight: float = 0.6  # 60%

    # Selection strategy
    strategy: str = "balanced"  # balanced, top_heavy, people_first, aesthetic_first

    # Temporal settings
    enforce_monthly_distribution: bool = True
    min_months_represented: int = 10

    # Diversity settings
    visual_diversity_threshold: float = 0.15
    enable_diversity_filter: bool = True

    # Quality thresholds
    min_combined_score: float = 30.0  # Reject below this
    prefer_with_faces: bool = True

    # Fallback behavior
    allow_duplicates_if_needed: bool = False
    year_month_tolerance: int = 1  # Include adjacent months
```

---

### Example Output

```json
{
  "year": 2024,
  "strategy": "balanced",
  "created_at": "2025-01-15T10:30:00Z",
  "stats": {
    "total_candidates": 1247,
    "avg_quality_score": 72.3,
    "avg_emotional_score": 45.8,
    "avg_combined_score": 56.6,
    "months_represented": 11,
    "photos_with_faces": 8
  },
  "photos": [
    {
      "path": "/Photos/2024/01/IMG_0234.jpg",
      "month": 1,
      "timestamp": "2024-01-15T14:30:00Z",
      "quality_score": 85.2,
      "emotional_score": 72.4,
      "combined_score": 77.5,
      "reason": "Birthday celebration - high emotional significance"
    },
    {
      "path": "/Photos/2024/02/IMG_0891.jpg",
      "month": 2,
      "timestamp": "2024-02-14T18:00:00Z",
      "quality_score": 92.1,
      "emotional_score": 88.3,
      "combined_score": 89.9,
      "reason": "Valentine's dinner - excellent quality and emotion"
    },
    ...
  ]
}
```

---

### Dependencies

- **Blocked by:**
  - Feature 1.1 (Photo Quality Analyzer) - ✅ Complete
  - Feature 1.2 (Emotional Significance Detector) - ✅ Complete

- **Blocks:**
  - Feature 2.1 (Multi-Circle Organization) - needs per-circle curation
  - Feature 3.1 (Reflection Interface) - needs curated Twelve to display

- **Related:**
  - BatchProcessor, Cache - reuse for fast curation
  - EXIF metadata extraction - for timestamps
  - Visual diversity algorithms - histogram/dhash

---

### Implementation Phases

#### Phase 1: Core Curation (MVP)
- Combined scoring (quality + emotional)
- Temporal grouping by month
- Basic selection (top N per month)
- TwelveSelection output
- Tests on synthetic data

#### Phase 2: Diversity Filters
- Visual similarity detection
- Duplicate removal
- Variety enforcement
- Tests on real photos

#### Phase 3: Advanced Strategies
- Multiple curation strategies
- Configuration presets
- Fallback handling
- User customization

---

### Definition of Done

- [ ] Feature specification complete
- [ ] User stories created
- [ ] Architecture designed
- [ ] Core curation engine implemented
- [ ] Temporal diversity working
- [ ] Visual diversity working
- [ ] Unit tests (>90% coverage)
- [ ] Integration tests on real libraries
- [ ] Performance benchmarks met (<30s for 1000 photos)
- [ ] Demo script with visualization
- [ ] Documentation complete

---

### Design Principles

**"Better to Skip Than Repeat"**
- If visual diversity can't be maintained, select fewer photos
- Quality over quantity within the Twelve

**"Temporal Spread Beats Absolute Best"**
- A good photo from an underrepresented month > amazing photo from overrepresented month
- Exception: If quality gap is massive (>30 points)

**"People Make Memories"**
- Photos with faces weighted higher by default
- Landscapes included for variety, not majority

**"Automate the Hard Choice"**
- Users shouldn't need to configure much
- Sensible defaults, advanced options available

---

### Metadata & Change History

| Version | Date       | Author | Changes                                     |
| ------- | ---------- | ------ | ------------------------------------------- |
| v1.0    | 2025-10-11 | Claude | Initial feature spec for Twelve Curation Engine |

