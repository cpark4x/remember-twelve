# Photo Quality Analyzer - Architecture

## Module Structure

```
photo_quality_analyzer/
│
├── __init__.py                      # Public API exports
│   └── Exports: PhotoQualityAnalyzer, QualityScore, config functions
│
├── analyzer.py                      # Main Orchestration
│   ├── PhotoQualityAnalyzer         # Main class
│   │   ├── analyze_photo()          # From file path
│   │   ├── analyze_image()          # From numpy array
│   │   ├── analyze_batch()          # Multiple photos
│   │   └── _preprocess_image()      # Resize/optimize
│   └── analyze_photo_simple()       # Convenience function
│
├── config.py                        # Configuration System
│   ├── QualityAnalyzerConfig        # Master config
│   ├── WeightsConfig                # Score weights
│   ├── ThresholdsConfig             # Quality tiers
│   ├── SharpnessConfig              # Algorithm params
│   ├── ExposureConfig               # Algorithm params
│   ├── PerformanceConfig            # Optimization
│   └── Preset functions             # Pre-configured setups
│
└── metrics/                         # Quality Metrics
    ├── __init__.py                  # Metrics exports
    │
    ├── sharpness.py                 # Blur Detection
    │   ├── calculate_sharpness_score()
    │   ├── get_sharpness_tier()
    │   └── calculate_sharpness_with_metadata()
    │
    ├── exposure.py                  # Exposure Analysis
    │   ├── calculate_exposure_score()
    │   ├── get_exposure_tier()
    │   ├── analyze_histogram()
    │   └── detect_exposure_issues()
    │
    └── composite.py                 # Score Combining
        ├── QualityScore             # Data container
        ├── calculate_quality_score()
        ├── create_quality_score()
        ├── get_quality_tier()
        ├── compare_scores()
        └── batch_calculate_scores()
```

## Data Flow

```
User Input (photo path)
        ↓
┌───────────────────────────────────────┐
│   PhotoQualityAnalyzer                │
│   (analyzer.py)                       │
├───────────────────────────────────────┤
│ 1. Load image from file               │
│ 2. Preprocess (resize to 1024px)      │
│ 3. Convert to numpy array             │
└───────────────┬───────────────────────┘
                ↓
        ┌───────┴───────┐
        ↓               ↓
┌────────────────┐  ┌────────────────┐
│ Sharpness      │  │ Exposure       │
│ (sharpness.py) │  │ (exposure.py)  │
├────────────────┤  ├────────────────┤
│ - Grayscale    │  │ - Grayscale    │
│ - Laplacian    │  │ - Histogram    │
│ - Variance     │  │ - Clipping     │
│ - Normalize    │  │ - Crushing     │
└────────┬───────┘  └────────┬───────┘
         │                   │
         │   Score: 0-100    │   Score: 0-100
         └─────────┬─────────┘
                   ↓
        ┌──────────────────────┐
        │ Composite            │
        │ (composite.py)       │
        ├──────────────────────┤
        │ - Weight scores      │
        │ - Calculate tier     │
        │ - Create QualityScore│
        └──────────┬───────────┘
                   ↓
            ┌──────────────┐
            │ QualityScore │
            ├──────────────┤
            │ sharpness    │
            │ exposure     │
            │ composite    │
            │ tier         │
            └──────────────┘
                   ↓
            Return to user
```

## Module Dependencies

```
analyzer.py
    ├── depends on → metrics.sharpness
    ├── depends on → metrics.exposure
    ├── depends on → metrics.composite
    └── depends on → config

metrics/composite.py
    └── NO dependencies (pure function)

metrics/sharpness.py
    └── depends on → opencv, numpy (external only)

metrics/exposure.py
    └── depends on → opencv, numpy (external only)

config.py
    └── NO dependencies (pure configuration)
```

**Key Design Feature:** Metrics have NO cross-dependencies. They can be used independently or replaced without affecting other modules.

## Interfaces

### Public API (from __init__.py)

```python
from photo_quality_analyzer import (
    # Main analyzer
    PhotoQualityAnalyzer,
    analyze_photo_simple,
    
    # Data types
    QualityScore,
    create_quality_score,
    
    # Configuration
    QualityAnalyzerConfig,
    get_default_config,
    create_custom_config,
    get_conservative_config,
    get_permissive_config,
)
```

### Metric Functions

```python
# Sharpness
calculate_sharpness_score(image: np.ndarray) -> float

# Exposure
calculate_exposure_score(image: np.ndarray) -> float

# Composite
calculate_quality_score(
    sharpness: float,
    exposure: float,
    weights: Optional[Dict[str, float]] = None
) -> float
```

## Design Patterns

### 1. Strategy Pattern
- Different metrics (sharpness, exposure) implement common interface
- Can swap or add metrics without changing analyzer

### 2. Facade Pattern
- `PhotoQualityAnalyzer` provides simple interface hiding complexity
- Users don't need to know about internal metric calculations

### 3. Configuration Pattern
- Centralized configuration with validation
- Preset configurations for common use cases
- Easy to extend with new parameters

### 4. Pure Functions
- Metrics are pure functions (same input → same output)
- No side effects, easy to test
- Composable and reusable

## Extension Points

### Add New Metric (Phase 2: Composition)

```python
# 1. Create new metric module
# metrics/composition.py
def calculate_composition_score(image: np.ndarray) -> float:
    # Implement composition scoring
    pass

# 2. Update config.py
class WeightsConfig:
    sharpness: float = 0.5
    exposure: float = 0.3
    composition: float = 0.2  # Add new weight

# 3. Update analyzer.py
def analyze_image(self, image: np.ndarray) -> QualityScore:
    sharpness = calculate_sharpness_score(image)
    exposure = calculate_exposure_score(image)
    composition = calculate_composition_score(image)  # Add calculation
    
    score = create_quality_score(
        sharpness, exposure, composition, weights=...
    )
```

### Add Custom Preprocessing

```python
class PhotoQualityAnalyzer:
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Resize (existing)
        image = self._resize_if_needed(image)
        
        # Add new preprocessing
        image = self._denoise(image)  # Custom
        image = self._enhance_contrast(image)  # Custom
        
        return image
```

### Add Database Storage (Phase 2)

```python
# Create new module: storage/db_handler.py
class QualityScoreDB:
    def save_score(self, photo_id: str, score: QualityScore):
        pass
    
    def get_score(self, photo_id: str) -> Optional[QualityScore]:
        pass

# Integrate in analyzer.py
def analyze_photo(self, photo_path: str) -> QualityScore:
    score = self._analyze(photo_path)
    
    if self.config.cache.enable_cache:
        self.db.save_score(photo_path, score)  # Save to DB
    
    return score
```

## Testing Strategy

### Unit Tests (Independent)

```
test_sharpness.py
    └── Tests sharpness.py in isolation

test_exposure.py
    └── Tests exposure.py in isolation

test_composite.py
    └── Tests composite.py in isolation

test_config.py
    └── Tests config.py validation
```

### Integration Tests

```
test_analyzer.py
    └── Tests full pipeline:
        - File loading
        - Preprocessing
        - Metric coordination
        - Score creation
        - Error handling
```

## Performance Characteristics

### Time Complexity

- **Sharpness:** O(n) where n = pixels (single Laplacian pass)
- **Exposure:** O(n) where n = pixels (single histogram pass)
- **Composite:** O(1) (simple arithmetic)
- **Overall:** O(n) linear with image size

### Space Complexity

- **Memory:** O(1) constant (images resized to max 1024px)
- **Preprocessing:** Creates one resized copy
- **Metrics:** Work on grayscale (1 channel vs 3)

### Optimization Techniques

1. **Image Resizing:** 1024px max → 10x faster
2. **Grayscale Conversion:** 3x less data to process
3. **Single-Pass Algorithms:** No iteration over data
4. **Efficient Data Types:** NumPy arrays for speed

## Error Handling Strategy

### Input Validation
- Type checking (numpy array vs other types)
- Dimension checking (2D or 3D arrays)
- Value range checking (scores 0-100)

### Graceful Degradation
- Batch processing continues on individual failures
- Returns None for failed photos
- Logs warnings but doesn't stop execution

### Configuration Validation
- Validates weights sum to 1.0
- Validates threshold ordering
- Raises ValueError with clear messages

## Future Architecture (Phase 2+)

```
photo_quality_analyzer/
├── analyzer.py              # Existing
├── config.py                # Existing
├── metrics/                 # Existing
│   ├── sharpness.py
│   ├── exposure.py
│   ├── composite.py
│   └── composition.py       # NEW: Phase 2
├── processors/              # NEW: Phase 2
│   ├── image_loader.py      # Load from S3/local
│   └── batch_processor.py   # Parallel processing
├── storage/                 # NEW: Phase 2
│   ├── db_handler.py        # Database operations
│   └── cache.py             # Score caching
└── utils/                   # NEW: Phase 2
    └── validation.py        # Input validation
```

---

**Design Principles:**
- Modular: Each component has single responsibility
- Testable: Pure functions with clear interfaces
- Extensible: Easy to add new metrics or features
- Performant: Optimized for speed without sacrificing accuracy
- Maintainable: Clear structure and comprehensive documentation
