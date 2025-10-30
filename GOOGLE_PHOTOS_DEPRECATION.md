# Google Photos API Deprecation Notice

## What Happened

On **March 31, 2025**, Google deprecated and removed the `photoslibrary.readonly` OAuth scope that allowed third-party applications to automatically access user photo libraries.

This means the Google Photos integration in Remember Twelve **no longer works** and cannot be fixed through configuration changes.

## Why It Fails

- OAuth authentication completes successfully ✅
- Tokens are issued and validate ✅
- But all API calls return `403 PERMISSION_DENIED` ❌

**This is intentional by Google**, not a bug in Remember Twelve.

## Google's Replacement: Picker API

Google now requires apps to use the "Picker API" which:
- Forces users to manually select photos via a UI
- Requires clicking through a pickerUri link or scanning QR code
- **Completely breaks automated curation** - the core feature of Remember Twelve

The Picker API is fundamentally incompatible with Remember Twelve's goal of automatically curating an entire year of photos.

## The Solution: Use Local Photo Sources

Remember Twelve was designed to be **source-agnostic**. The local file source already works perfectly and provides the same curation experience.

### Option A: Google Takeout (Recommended)

**One-time manual export, then full automation:**

1. Go to https://takeout.google.com
2. Click "Deselect all"
3. Select only "Google Photos"
4. Click "All photo albums included" → "Multiple formats" → Select specific year (e.g., 2023)
5. Click "Next step" → Choose file size and delivery method
6. Download the export (this may take hours for large libraries)
7. Extract the ZIP file
8. Run Remember Twelve pointing to the extracted folder

The photos include rich metadata JSON files from Google with dates, locations, etc.

### Option B: Google Drive Desktop Sync

**Always current, automatic sync:**

1. Install Google Drive desktop application
2. In settings, enable "Google Photos" sync
3. Let it sync your photo library locally
4. Point Remember Twelve to the synced Photos folder
5. Photos stay current automatically

### Option C: Direct Download

**Simple but manual:**

1. Use Google Photos web interface
2. Select photos from your target year
3. Download them (⋮ menu → Download)
4. Organize in a folder
5. Point Remember Twelve to the folder

## What Changed in Remember Twelve

- The `google_photos_source.py` integration is **deprecated** (non-functional)
- The local file source (`file_system_source.py`) is the **primary supported method**
- OAuth credentials and Google Photos API dependencies remain in code for reference but are non-functional

## Why We Can't Work Around This

- Google explicitly deprecated the scope at the API level
- There's no "configuration fix" - the API endpoint rejects all calls
- Attempting to use deprecated scopes results in 403 errors
- The Picker API requires manual user interaction, breaking automation
- **This is a permanent change by Google, not temporary**

## Impact on Users

If you were planning to use Google Photos integration:
- Use Google Takeout workflow (Option A above)
- The curation AI works identically with local files
- You get the same "Remember Twelve" output
- Only the photo acquisition method changed

## Benefits of Local Sources

- **No API rate limits** - process as fast as disk allows
- **No OAuth complexity** - just point to a folder
- **Full metadata** - EXIF data, filenames, folder structure
- **Privacy** - photos never leave your machine except when you choose
- **Reliable** - no dependency on external API changes
- **Multi-source** - works with any photo collection (iCloud, Amazon Photos, camera roll, etc.)

## Future Direction

Remember Twelve will focus on being the **best local photo curator**, supporting:
- Local photo libraries from any source
- Rich EXIF metadata parsing
- Flexible folder structures
- Multiple input sources in one curation

The removal of Google Photos API access actually aligns with Remember Twelve's philosophy: **your photos, your control, local AI curation**.
