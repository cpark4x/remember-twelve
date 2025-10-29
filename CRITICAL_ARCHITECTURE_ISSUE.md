# CRITICAL: Google Photos Curation Architecture Issue

**Status**: BLOCKING - Prevents v1.0 Google Photos Integration
**Severity**: High - Makes `/api/curate` endpoint unusable with real Google Photos libraries
**Discovered**: 2025-10-29 during end-to-end pipeline testing

## The Problem

The integrated curation pipeline (`POST /api/curate`) is **fundamentally inefficient** for Google Photos because it downloads ALL photos from a year before selecting the best 12.

###Test Results:
- Called `/api/curate` for year 2023
- Process ran for 4+ minutes without completing
- Server logs show it started curation but never finished
- Root cause: Downloading hundreds of photos from Google Photos (5-30+ minutes)

## Root Cause Analysis

### Current Architecture (BROKEN)

The `PhotoSource` interface was designed for local file access:

```python
class PhotoSource(ABC):
    def scan(self, year: int) -> Iterator[str]:
        """Yields absolute file paths to photos"""
        pass
```

**Problem**: Google Photos doesn't have "local file paths". The `GooglePhotosSource` implementation works around this by:
1. **Downloading every photo** from the year to a temp cache
2. Yielding the temp file paths
3. Curator analyzes downloaded files
4. Select best 12
5. Clean up temp files

**This means**: For a user with 300 photos from 2023, we download 300 photos (potentially gigabytes) just to select 12.

### Current Flow (INEFFICIENT)

```
User → POST /api/curate {year: 2023}
  ↓
CurationService.curate_year()
  ↓
PhotoSourceFactory.create_google_photos()
  ↓
TwelveCurator.curate_from_source()
  ↓
photo_source.scan(year=2023)  ← DOWNLOADS ALL 300 PHOTOS HERE (5-30+ mins)
  ↓
Analyze each photo (quality + emotional)
  ↓
Select best 12
  ↓
Save to database
  ↓
Clean up 300 downloaded files
```

**Time**: 5-30+ minutes
**Network**: Gigabytes of data
**User Experience**: Unusable

## Optimal Architecture (NEEDED)

### Two-Phase Approach

**Phase 1: Metadata-Only Selection** (Fast)
1. List all photo metadata from Google Photos (no downloads)
2. Score based on metadata: dimensions, creation date, file size
3. Pre-filter to top ~50 candidates

**Phase 2: Download & Deep Analysis** (Targeted)
1. Download only the ~50 candidates
2. Run deep analysis (quality + emotional scoring)
3. Select final 12
4. Save to database

### Proposed Interface Change

```python
class PhotoSource(ABC):
    def list_metadata(
        self,
        year: int
    ) -> Iterator[PhotoMetadata]:
        """
        List photo metadata without downloading.

        Returns:
            PhotoMetadata with:
            - id: Unique identifier
            - timestamp: Creation time
            - month: Month number (1-12)
            - width/height: Dimensions
            - file_size: Size in bytes
            - mime_type: Image format
            - source_url: Where to download from
        """
        pass

    def download_photo(
        self,
        photo_id: str,
        destination: Path
    ) -> bool:
        """Download a specific photo by ID"""
        pass
```

### Optimal Flow (FAST)

```
User → POST /api/curate {year: 2023}
  ↓
CurationService.curate_year()
  ↓
photo_source.list_metadata(year=2023)  ← LIST ONLY (fast, ~5 seconds)
  ↓
Pre-score based on metadata (dimensions, date, file size)
  ↓
Identify top ~50 candidates
  ↓
photo_source.download_photo() for each candidate  ← DOWNLOAD ~50 (30-60 seconds)
  ↓
Deep analysis (quality + emotional) on 50 photos
  ↓
Select best 12
  ↓
Save to database
```

**Time**: 30-90 seconds
**Network**: ~50 photos instead of 300
**User Experience**: Acceptable

## Impact Assessment

### What Works Now
✅ API endpoint structure (`POST /api/curate`)
✅ Google Photos authentication
✅ Photo quality analyzer
✅ Emotional significance analyzer
✅ Database persistence
✅ Service layer architecture

### What's Broken
❌ Curation takes 5-30+ minutes (should be <2 mins)
❌ Downloads ALL photos unnecessarily
❌ Wastes network bandwidth
❌ Wastes storage space
❌ Poor user experience
❌ Blocks v1.0 milestone

## Required Changes

### 1. Redesign PhotoSource Interface
- Add `list_metadata()` method (no downloads)
- Add explicit `download_photo(id, dest)` method
- Keep backward compatibility for local sources

### 2. Update GooglePhotosSource
- Implement `list_metadata()` using Google Photos API
- Return metadata without downloading
- Update `download_photo()` to download by ID

### 3. Update TwelveCurator
- Add two-phase curation:
  - Phase 1: Metadata-based pre-filtering
  - Phase 2: Deep analysis on candidates
- Add `curate_from_metadata()` method

### 4. Update CurationService
- Orchestrate two-phase flow
- Handle download errors gracefully
- Report progress to API caller

### 5. Test End-to-End
- Test with real 2023 library (300+ photos)
- Verify completion in <2 minutes
- Verify correct 12 photos selected

## Priority

**CRITICAL** - This blocks the core value proposition: effortless Google Photos curation.

## Estimated Effort

- Interface redesign: 1 hour
- GooglePhotosSource update: 2 hours
- Curator update: 3 hours
- Service layer update: 1 hour
- Testing: 2 hours

**Total**: ~9 hours (1-2 days)

## Next Steps

1. ✅ Document the issue (this file)
2. [ ] Design new PhotoSource interface
3. [ ] Implement metadata-only listing for GooglePhotosSource
4. [ ] Update curator for two-phase approach
5. [ ] Test end-to-end with real library
6. [ ] Verify <2 minute completion time
7. [ ] Update documentation

## Notes

- This issue was hidden during development because we tested with small datasets
- The architecture is sound for local photos (LocalPhotoSource)
- Only cloud sources (GooglePhotosSource) are affected
- The fix will make the system much more scalable
- Consider adding progress updates (e.g., WebSocket or SSE)

## Related Files

- [src/photo_sources/base.py](src/photo_sources/base.py) - PhotoSource interface
- [src/photo_sources/google_photos_source.py](src/photo_sources/google_photos_source.py) - Google Photos implementation
- [src/twelve_curator/curator.py](src/twelve_curator/curator.py) - Curation engine
- [src/services/curation_service.py](src/services/curation_service.py) - Service layer
- [src/api/routes/curations.py](src/api/routes/curations.py) - API endpoint
