# Google Takeout Workflow Guide

Complete guide for using Google Takeout to curate your Google Photos with Remember Twelve.

## Why Google Takeout?

Google deprecated their Photos Library API in March 2025. The only reliable way to access your full photo library is through Google Takeout, which provides:

- ✅ Complete photo library export
- ✅ Rich metadata (dates, locations, albums)
- ✅ One-time setup, then fully automated curation
- ✅ Works offline, no API rate limits
- ✅ Full privacy - photos stay local

## Step-by-Step Process

### 1. Request Your Google Takeout Export

1. **Go to Google Takeout**: https://takeout.google.com

2. **Deselect all products**:
   - Click "Deselect all" at the top

3. **Select only Google Photos**:
   - Scroll down and check ☑️ "Google Photos"

4. **Configure the export**:
   - Click "All photo albums included" button
   - Select "Multiple formats" tab
   - Choose specific years or date ranges
   - Example: Select only "2023" if you want photos from that year

5. **Choose delivery method**:
   - Click "Next step"
   - File size: 50GB recommended (or smaller for faster downloads)
   - Delivery method:
     - "Send download link via email" (easiest)
     - OR "Add to Drive" / "Add to Dropbox"
   - Click "Create export"

6. **Wait for export**:
   - Google will email you when ready
   - Can take minutes to hours depending on library size
   - You'll receive a download link

### 2. Download and Extract

1. **Download the ZIP file(s)**:
   - Click the download link in Google's email
   - May be split into multiple files (e.g., takeout-20231015-001.zip, takeout-20231015-002.zip)

2. **Extract the archives**:
   ```bash
   # If single file
   unzip takeout-*.zip -d ~/google-photos-export

   # If multiple parts, extract all
   cd ~/Downloads
   for file in takeout-*.zip; do
       unzip "$file" -d ~/google-photos-export
   done
   ```

3. **Locate your photos**:
   ```
   ~/google-photos-export/
   └── Takeout/
       └── Google Photos/
           ├── 2023-01-15.jpg
           ├── 2023-01-15.jpg.json  ← metadata
           ├── 2023-02-20.jpg
           ├── 2023-02-20.jpg.json
           └── ... (all your photos)
   ```

### 3. Run Remember Twelve

**Option A: Use the sync script**

```bash
cd remember-twelve
./sync_photos.sh ~/google-photos-export/Takeout/Google\ Photos
```

**Option B: Direct curation**

```bash
python3 curate_from_google_photos.py \
    --source ~/google-photos-export/Takeout/Google\ Photos \
    --year 2023 \
    --strategy balanced \
    --flexible-months \
    --output ui/photos_data.json
```

**Option C: Interactive mode**

```bash
# Run the curation pipeline
python3 -c "
from src.photo_sources.factory import PhotoSourceFactory
from src.curation.pipeline import CurationPipeline

# Create local file source
source = PhotoSourceFactory.create_local_filesystem(
    '~/google-photos-export/Takeout/Google Photos'
)

# Set up curation pipeline
pipeline = CurationPipeline(source)

# Curate year 2023
results = pipeline.curate(
    year=2023,
    target_count=12,
    strategy='balanced'
)

print(f'Curated {len(results)} photos!')
"
```

### 4. View Your Remember Twelve

```bash
# Start the web viewer
cd ui
python3 sync_server.py

# Open browser to: http://localhost:8765
```

## Understanding Takeout Metadata

Google Takeout includes JSON files with rich metadata:

```json
{
  "title": "IMG_1234.jpg",
  "description": "",
  "imageViews": "156",
  "creationTime": {
    "timestamp": "1672531200",
    "formatted": "Jan 1, 2023, 12:00:00 AM UTC"
  },
  "photoTakenTime": {
    "timestamp": "1672531200",
    "formatted": "Jan 1, 2023, 12:00:00 AM UTC"
  },
  "geoData": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude": 0.0,
    "latitudeSpan": 0.0,
    "longitudeSpan": 0.0
  },
  "geoDataExif": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude": 0.0,
    "latitudeSpan": 0.0,
    "longitudeSpan": 0.0
  }
}
```

Remember Twelve automatically parses this metadata for better curation.

## Tips & Best Practices

### Filtering by Year

When requesting your export in step 1:
- Select specific years to reduce download size
- Multiple years? Request separate exports or one combined
- Only need 2023? Select just that year

### Storage Considerations

- Full Google Photos library can be 100GB+
- Filter by year if storage limited
- Delete export after curation if space tight
- Keep export if you want to re-curate with different settings

### Re-Curating

Once you have the Takeout export:
```bash
# Try different strategies
./sync_photos.sh ~/google-photos-export --strategy diverse
./sync_photos.sh ~/google-photos-export --strategy temporal
./sync_photos.sh ~/google-photos-export --strategy quality

# Try different years
./sync_photos.sh ~/google-photos-export --year 2022
./sync_photos.sh ~/google-photos-export --year 2024
```

### Incremental Updates

For ongoing use:
1. **Option A**: Request new Takeout every few months
2. **Option B**: Use Google Drive sync (see below)

### Google Drive Sync Alternative

Instead of Takeout, sync continuously:

```bash
# 1. Install Google Drive desktop app
# 2. Enable Photos sync in settings
# 3. Point Remember Twelve to sync folder

./sync_photos.sh ~/GoogleDrive/Google\ Photos
```

This keeps photos always current but requires the Drive app running.

## Troubleshooting

### "No photos found for year 2023"

Check the folder structure:
```bash
ls -la ~/google-photos-export/Takeout/Google\ Photos/
```

Photos should be directly in this folder, not nested deeper.

### "Failed to parse metadata"

Some JSON files may be corrupted. Remember Twelve will skip them and continue.

### "Out of storage space"

- Request smaller export (by year)
- Delete export after curation
- Use external drive for Takeout

### "Takeout link expired"

Google Takeout links expire after ~7 days. Request a new export.

## Comparison: Takeout vs API

| Feature | Old API (Deprecated) | Google Takeout |
|---------|---------------------|----------------|
| Setup | OAuth config | One-time download |
| Speed | Rate limited | Local disk speed |
| Privacy | Photos accessed via API | Fully local |
| Reliability | Depends on Google | Works offline |
| Metadata | Basic | Rich JSON |
| Automation | Fully automated | Manual export, auto curation |
| Storage | Temporary cache | Requires disk space |

## Next Steps

After successfully curating with Takeout:

1. ✅ View your Remember Twelve in the web UI
2. ✅ Try different curation strategies
3. ✅ Export to different formats
4. ✅ Set up regular Takeout exports for new photos

Remember: The AI curation is identical whether photos come from API or Takeout. You get the same quality Remember Twelve experience!
