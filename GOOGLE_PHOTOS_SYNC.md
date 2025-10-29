# Google Photos Sync Feature

## Overview

Remember-Twelve now supports direct synchronization with your Google Photos library, automatically curating and displaying your best 12 photos of the year with flexible month distribution.

---

## Features

✅ **One-Click Sync** - Authenticate once, sync anytime
✅ **Flexible Month Assignment** - Photos fill empty months intelligently
✅ **AI-Powered Curation** - Quality + emotional significance analysis
✅ **Privacy-First** - Photos cached locally, no permanent storage
✅ **Smart Distribution** - Photos placed in capture months, others fill gaps

---

## Quick Start

### Option 1: Command Line (Simple)

```bash
# Sync photos from 2023 with flexible month distribution
./sync_photos.sh 2023
```

Then open `ui/viewer.html` in your browser.

### Option 2: Web Interface (Best UX)

```bash
# Terminal 1: Start photo server
cd ui && python3 photo_server.py

# Terminal 2: Start sync server
python3 ui/sync_server.py

# Browser: Open http://localhost:8080/viewer_dynamic.html
# Click "Sync with Google Photos" button
```

---

## How It Works

### Architecture

```
Google Photos API
  ↓
GooglePhotosClient (OAuth + Fetch)
  ↓
TwelveCurator (AI Analysis + Distribution)
  ↓
Flexible Month Assignment
  ↓
ui/photos_data.json + ui/photos/
  ↓
Web Viewer Display
```

### Photo Distribution Logic

**Priority Fill with Smart Proximity:**

1. **Phase 1: Exact Match** - Place photos in their capture months
   - February photo → February slot
   - September photo → September slot

2. **Phase 2: Best Available** - Fill empty months with remaining photos
   - Sort remaining photos by combined score
   - Distribute to empty months in order

**Example:**
- You have 8 photos: Feb(1), Sep(6), Dec(1)
- **Result:** Feb slot gets Feb photo, Sep slot gets best Sep photo, Dec slot gets Dec photo
- Remaining 5 Sep photos → distributed to Jan, Mar, Apr, May, Jun (by score)

---

## Setup (First Time Only)

### 1. Google Cloud Project Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project: "Remember Twelve"
3. Enable **Photos Library API**
4. Create OAuth 2.0 credentials:
   - Application type: Desktop app
   - Name: "Remember Twelve Desktop"
5. Download credentials as `google_photos_credentials.json`
6. Place in project root directory

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. First Authentication

```bash
# Run sync for first time
./sync_photos.sh 2023

# Browser will open for Google OAuth
# Grant permissions: "See and download your Google Photos library"
# Token saved to ~/.remember_twelve/tokens.db
```

---

## Usage

### CLI Options

```bash
# Basic sync with flexible months
python curate_from_google_photos.py --year 2023 --flexible-months

# Custom curation strategy
python curate_from_google_photos.py \
  --year 2024 \
  --strategy people_first \
  --flexible-months

# Limit photos for testing
python curate_from_google_photos.py \
  --year 2023 \
  --limit 50 \
  --flexible-months
```

### Strategies

- **balanced** (default) - Balances quality, emotion, and diversity
- **aesthetic_first** - Prioritizes visual quality
- **people_first** - Prioritizes photos with faces
- **top_heavy** - Selects only highest-scoring photos

### Flags

- `--flexible-months` - **NEW!** Fills empty months with photos from other months
- `--year` - Year to curate (required)
- `--strategy` - Curation strategy (default: balanced)
- `--limit` - Limit number of photos to analyze (for testing)
- `--output` - Custom output JSON file path

---

## Web Viewer Integration

### Sync Button

The web viewer now includes a "Sync with Google Photos" button in the header.

**Location:** `ui/viewer_dynamic.html` header section

**Behavior:**
1. Click "Sync with Google Photos"
2. Button shows "Syncing..." with rotating icon
3. Status updates: "Fetching photos from Google Photos..."
4. On success: "Sync complete! Reloading..."
5. Gallery reloads with new photos

### Endpoints

**Sync Server (`ui/sync_server.py`):**

- **POST /sync** - Trigger sync for specified year
  ```json
  {
    "year": 2023
  }
  ```

- **GET /status** - Check current photo count
  ```json
  {
    "photo_count": 12,
    "photos_exist": true
  }
  ```

---

## Edge Cases

### Fewer than 12 Photos

**Example:** User has 3 photos from 2023

**Behavior:**
- Places 3 photos in their capture months
- Leaves 9 months empty with subtle placeholders
- No fake data or padding

**UI:** Empty months show month name + em dash (—)

### More than 12 Photos

**Example:** User has 50 photos, all from vacation in September

**Behavior:**
- Curator selects best 12 photos (highest quality + emotional scores)
- Places best September photo in September slot
- Distributes remaining 11 photos across empty months
- Photos maintain metadata showing original capture date

**UI:** Can show "Aug photo" annotation when displayed in different month

### No Photos for Selected Year

**Behavior:**
- CLI shows error: "No photos found in 2023"
- Suggests trying different year
- No output file generated

### All Photos Same Month

**Example:** 12 vacation photos from August

**Behavior:**
- Places best photo in August slot
- Distributes remaining 11 across other months
- Maintains temporal proximity where possible

---

## Security & Privacy

### Token Storage

- **Location:** `~/.remember_twelve/tokens.db`
- **Permissions:** User-only (600)
- **Encryption:** File-level permissions (single-user app)
- **Auto-refresh:** Tokens refresh automatically when expired

### Photo Storage

- **Cache:** Temporary local storage in `ui/photos/`
- **Duration:** Until next sync or manual cleanup
- **Persistence:** No permanent cloud storage
- **Privacy:** Photos never leave your machine

### API Rate Limits

- **Google Photos:** 10,000 requests/day per project
- **Typical sync:** ~10-20 API calls for 100 photos
- **Well within limits** for personal use

### OAuth Scopes

- **Requested:** `photoslibrary.readonly` (read-only access)
- **Never:** Write, delete, or modify permissions
- **Revocation:** [Google Account Permissions](https://myaccount.google.com/permissions)

---

## Troubleshooting

### "Credentials file not found"

**Solution:**
```bash
# Ensure google_photos_credentials.json exists
ls google_photos_credentials.json

# If missing, download from Google Cloud Console
```

### "Invalid credentials"

**Solution:**
```bash
# Delete existing token and re-authenticate
rm ~/.remember_twelve/tokens.db
./sync_photos.sh 2023
```

### "Rate limit exceeded"

**Solution:**
- Wait 24 hours (daily quota resets at midnight PST)
- Use `--limit` flag to reduce API calls during testing

### "No photos found"

**Possible causes:**
1. Wrong year specified
2. No photos in Google Photos library for that year
3. OAuth permissions not granted

**Solution:**
```bash
# Check year range
python curate_from_google_photos.py --year 2024 --flexible-months

# Verify permissions at myaccount.google.com/permissions
```

### Sync button not working

**Solution:**
```bash
# Ensure sync server is running
python3 ui/sync_server.py

# Check endpoint
curl http://localhost:5002/status
```

---

## Development

### Project Structure

```
remember-twelve/
├── src/
│   ├── photo_sources/
│   │   ├── google_photos_client.py     # OAuth + API wrapper
│   │   └── token_manager.py            # Secure token storage
│   └── twelve_curator/
│       └── curator.py                   # NEW: distribute_to_twelve_months()
├── ui/
│   ├── viewer_dynamic.html              # NEW: Sync button + UI
│   ├── photo_server.py                  # Serves photos locally
│   └── sync_server.py                   # NEW: Web sync endpoint
├── curate_from_google_photos.py         # UPDATED: --flexible-months flag
├── sync_photos.sh                       # NEW: One-command sync script
└── google_photos_credentials.json       # YOUR OAuth credentials
```

### Testing

```bash
# Test flexible distribution with local photos
python curate_from_google_photos.py \
  --year 2023 \
  --limit 8 \
  --flexible-months

# Verify output
cat ui/photos_data.json

# Test sync server
python ui/sync_server.py &
curl -X POST http://localhost:5002/sync \
  -H "Content-Type: application/json" \
  -d '{"year": 2023}'
```

---

## Future Enhancements

**Phase 3 (Optional):**
- [ ] Real-time progress updates during sync
- [ ] Photo count preview before sync
- [ ] Manual month reassignment UI
- [ ] "Aug photo in Jul slot" annotations

**Phase 4 (Advanced):**
- [ ] Multi-year support (2022, 2023, 2024 tabs)
- [ ] Custom curation preferences (emotion weights)
- [ ] Export to PDF/slideshow
- [ ] Cloud deployment for multi-user access

---

## Support

**Documentation:**
- [Google Photos API Docs](https://developers.google.com/photos/library/guides/overview)
- [Feature Spec](2-product/Features/epic-1-ai-curation/feature-1.5-google-photos-integration.md)
- [Design Doc](3-design/GooglePhotosIntegration.md)

**Issues:**
- Report bugs or request features in project issues
- Include sync logs and error messages

---

## Philosophy

**Why flexible month assignment?**

Remember-Twelve embraces the principle of **"Preservation Over Perfection"** (see `1-vision/Principles.md`). Life doesn't distribute evenly across 12 months. Some months are full of moments, others are quiet. Rather than forcing perfection, we:

1. **Honor actual distribution** - Place photos where they belong
2. **Fill gaps intelligently** - Use best available photos for empty months
3. **Embrace incompleteness** - If only 3 photos exist, show 3 (not 12 fake ones)
4. **Maintain transparency** - Users can see original capture dates

**Result:** A year view that's both complete (12 months shown) and honest (reflects your actual life).

---

**Enjoy your synced photo memories!** 📸✨
