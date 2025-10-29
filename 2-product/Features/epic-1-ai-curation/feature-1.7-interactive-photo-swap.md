# Feature 1.7: Interactive Photo Swap

**Status:** ✅ COMPLETED
**Epic:** Epic 1 - AI-Powered Photo Curation Engine
**Completion Date:** October 29, 2025

---

## Overview

Interactive Photo Swap allows users to manually override AI-curated photo selections for any month in their Remember Twelve calendar. Users can choose from two modes: "Auto Pick" (AI-suggested alternatives from the same capture month) or "Browse All" (complete freedom to choose any photo from their collection).

---

## Problem Statement

While AI curation provides excellent automated photo selection, users need the ability to:
- Replace photos they don't like with alternatives
- Choose personal favorites over AI selections
- Exercise creative control over their year-in-review
- Fix cases where AI misses context (e.g., meaningful but technically lower-quality photos)

Without a swap mechanism, users are locked into AI choices and may feel the tool lacks flexibility.

---

## User Stories

### Primary User Stories

**As a** busy parent
**I want to** quickly swap a photo I don't like with a better alternative
**So that** my Remember Twelve truly represents my favorite memories without spending hours curating

**As a** memory keeper
**I want to** see all photos from the same month when swapping
**So that** I can choose the best alternative based on my personal preferences

**As a** creative user
**I want to** choose any photo from any month for any calendar slot
**So that** I have complete flexibility in how I represent my year

### Edge Cases Handled

- Empty months with no photos
- Months with only one photo (no alternatives to show)
- HEIC format photos converted to browser-compatible JPG
- Duplicate photos removed from alternatives
- Photos with missing EXIF data handled gracefully

---

## Solution Design

### User Flow

```
1. User views Remember Twelve calendar (12 months displayed)
2. User hovers over any photo → "Swap Photo" button appears
3. User clicks "Swap Photo" → Modal opens
4. Modal shows two tabs:
   - "Auto Pick" (default): 5 best alternatives from same capture month
   - "Browse All": All 35 photos sorted by quality score
5. User clicks any alternative photo → Photo swaps immediately
6. Modal closes, calendar updates with new selection
7. User's choice saved in browser localStorage (persists across sessions)
```

### Technical Architecture

**Frontend Components:**
- Hover overlay with swap button on each photo card
- Modal with tabbed interface (Auto Pick / Browse All)
- Alternative photo grid with thumbnail previews
- Click handlers for photo selection and swapping

**Data Structure:**
```json
{
  "photos": [...],  // All 35 analyzed photos
  "month_distribution": {
    "January": {...},  // Currently selected photo per month
    ...
  },
  "photos_by_capture_month": {
    "January": [...],  // All photos taken in January
    ...
  },
  "user_swaps": {
    "March": "/path/to/photo.jpg"  // Track manual selections
  }
}
```

**Persistence:**
- User swap preferences stored in browser localStorage
- Loaded automatically on page refresh
- Survives browser restarts

---

## Implementation Details

### What Was Built

**1. Backend (Python Curation Script)**
- Extended `curate_local_photos.py` to save all candidate photos
- Added `photos_by_capture_month` grouping for Auto Pick alternatives
- Added `user_swaps` tracking object
- Quality scoring algorithm: resolution (50%) + file size (30%) + aspect ratio (20%)
- Emotional scoring: base 50 + size bonus + resolution bonus

**2. Frontend (HTML/CSS/JavaScript)**
- Hover overlay CSS with fade-in animation
- Modal with 2-tab interface (Auto Pick / Browse All)
- Swap button styling (Maru Coffee aesthetic)
- Alternative photo grid (responsive, 180px thumbnails)
- Photo swap JavaScript function
- localStorage persistence layer

**3. UI/UX Polish**
- Smooth animations (350ms transitions)
- Click outside modal to close
- Escape key to close (not implemented but easy to add)
- Visual feedback on hover
- Score badges on alternative photos
- Calendar month labels made more prominent (16px, bold)

### Files Modified

**Core Files:**
- `curate_local_photos.py` - Added grouping and tracking logic
- `ui/viewer_dynamic.html` - Added swap UI and functionality

**Supporting Files:**
- `ui/twelve_2023_balanced.json` - Updated with new data structure
- `ui/photos/*.jpg` - HEIC conversion for browser compatibility

---

## Testing Performed

### Manual Testing Completed

✅ **Basic Swap Functionality**
- Hover over photo → Swap button appears
- Click "Swap Photo" → Modal opens
- Both tabs load correctly
- Photos display in both tabs
- Click alternative → Photo swaps
- Modal closes after swap

✅ **Auto Pick Tab**
- Shows alternatives from same capture month
- Filters out currently selected photo
- Shows up to 5 alternatives
- Handles empty months gracefully
- Displays "Try Browse All" message when no alternatives

✅ **Browse All Tab**
- Shows all 35 photos
- Photos sorted by quality score (high to low)
- All photos clickable
- Any photo can be selected for any month

✅ **Persistence**
- Swap preferences saved to localStorage
- Preferences load on page refresh
- Multiple swaps tracked correctly

✅ **Edge Cases**
- HEIC photos converted to JPG ✅
- Duplicate photos removed ✅
- Photos with missing EXIF handled ✅
- Calendar displays all 12 months in order ✅
- Empty months show em dash (—) ✅

✅ **Browser Compatibility**
- Chrome: Works ✅
- Safari: Works (HEIC conversion required) ✅
- Firefox: Not tested yet

✅ **Aesthetic & UX**
- Maru Coffee design system applied ✅
- Month labels prominent and readable ✅
- Smooth animations ✅
- Responsive layout ✅

---

## Success Metrics

### Immediate Success Indicators
- ✅ Feature works without errors
- ✅ Users can swap photos in ≤3 clicks
- ✅ Swaps persist across page refreshes
- ✅ All 35 photos accessible for swapping

### Future Tracking (When Production)
- **Swap Rate**: % of users who use swap feature
- **Swaps Per User**: Average number of swaps per session
- **Auto Pick vs. Browse All**: Which tab gets more usage
- **Swap Satisfaction**: Post-swap survey rating

---

## Known Limitations & Future Improvements

### Current Limitations
- Swaps stored only in browser localStorage (not synced across devices)
- No undo functionality for swaps
- No bulk swap capability
- Cannot preview photo in full size before swapping
- No keyboard shortcuts (arrow keys, ESC)

### Planned Improvements (Future)
- **Sync swaps to backend** - Persist across devices
- **Undo/redo functionality** - Revert recent swaps
- **Photo preview mode** - Full-size view before swapping
- **Keyboard navigation** - Arrow keys to browse, ESC to close
- **Batch operations** - "Replace all low-quality photos"
- **Smart suggestions** - "These 3 photos are similar, keep only one?"
- **Photo comparison view** - Side-by-side before/after
- **Favorites/pins** - Pin certain photos to always include

---

## User Documentation

### How to Swap a Photo

1. **Open Remember Twelve viewer** at `http://localhost:8080/viewer_dynamic.html`
2. **Hover over any photo** in the calendar
3. **Click "Swap Photo"** button that appears
4. **Choose your replacement:**
   - **Auto Pick tab**: Best alternatives from the same month
   - **Browse All tab**: Choose from all 35 photos
5. **Click any alternative photo** to replace the current one
6. **Done!** Photo swaps immediately and preference is saved

### Tips
- Auto Pick is fastest for quick replacements
- Browse All gives you complete creative control
- Your choices are automatically saved
- You can swap as many times as you want

---

## Technical Debt & Maintenance Notes

### Code Quality
- ✅ Clean separation of concerns (data, UI, logic)
- ✅ Console logging for debugging (can be removed in production)
- ⚠️ Some duplication between Auto Pick and Browse All rendering
- ⚠️ Modal HTML embedded in main HTML file (could extract to component)

### Performance Considerations
- ✅ Lazy loading for Browse All tab
- ✅ Efficient photo indexing with `findIndex()`
- ⚠️ All 35 photos load when modal opens (pre-loading strategy)
- ⚠️ No image lazy loading on thumbnails (could add for large collections)

### Browser Compatibility
- ✅ Modern JavaScript (ES6+) used throughout
- ✅ LocalStorage API (supported by all modern browsers)
- ⚠️ No fallback for browsers without localStorage
- ⚠️ HEIC support requires conversion (handled)

---

## Deployment Notes

### Prerequisites
- Python 3.x with PIL (Pillow) and pillow-heif installed
- Photo server running (`photo_server.py`)
- Photos in `~/Pictures/2023 Remember Twelve/` directory
- Curated data in `ui/twelve_2023_balanced.json`

### How to Run
```bash
# 1. Run curation (if new photos added)
python3 curate_local_photos.py

# 2. Copy data to UI directory
cp ui/photos_data.json ui/twelve_2023_balanced.json

# 3. Start photo server
cd ui && python3 photo_server.py

# 4. Open in browser
open http://localhost:8080/viewer_dynamic.html
```

### How to Deploy
1. Commit all changes to git
2. Push to GitHub repository
3. (Future) Deploy to hosting service with static file serving

---

## Changelog

| Version | Date       | Changes                                      |
| ------- | ---------- | -------------------------------------------- |
| v1.0    | 2025-10-29 | Initial implementation completed             |
| v1.0.1  | 2025-10-29 | Fixed Auto Pick to use capture month         |
| v1.0.2  | 2025-10-29 | Changed Browse All to show all 35 photos     |
| v1.0.3  | 2025-10-29 | Added pre-loading when modal opens           |

---

## Related Documentation

- [Epic 1: AI-Powered Photo Curation Engine](../../Epics/epic-1-ai-curation-engine.md)
- [AESTHETIC-GUIDE.md](.design/AESTHETIC-GUIDE.md) - Maru Coffee design system
- [README.md](../../README.md) - Project overview

---

**Author:** Chris Park
**Reviewed By:** N/A (Solo project)
**Last Updated:** October 29, 2025
