# Apple Photos Integration Guide

Complete guide for using Apple Photos with Remember Twelve.

## Why Apple Photos Works Great

Unlike Google Photos (which deprecated their API), Apple Photos provides:

- ✅ **Native file export** - Built into Photos app
- ✅ **Fast and local** - No waiting for cloud processing
- ✅ **Rich metadata** - GPS, faces, albums, favorites preserved
- ✅ **Simple workflow** - 2 minutes to export, immediate curation
- ✅ **No cloud dependency** - Works offline

## Quick Start (3 Methods)

### Method A: Automated Export (macOS only) 🚀 EASIEST

If you're on macOS, Remember Twelve can automatically export from Photos:

```bash
# Export and curate your entire 2023 library
./sync_photos.sh --apple-photos 2023

# Export specific album
./sync_photos.sh --apple-photos 2023 --album "Favorites"
```

The script will:
1. 🔓 Request permission to access Photos (one-time)
2. 📤 Export photos from specified year
3. 🤖 Run AI curation automatically
4. 🎉 Show your Remember Twelve!

**First-time setup**: macOS will ask for Photos access permission. Click "OK" to allow.

### Method B: Manual Export (All platforms) 📋 UNIVERSAL

Works on macOS, Windows (with iCloud Photos), or any platform:

#### Step 1: Export from Photos App

**On macOS:**
1. Open Photos app
2. Click "Photos" in sidebar
3. Filter by year:
   - Click search icon (🔍)
   - Type: `date:2023`
   - Press Enter
4. Select all photos:
   - `Cmd + A` (or Edit → Select All)
5. Export:
   - File → Export → Export Unmodified Originals
   - Choose destination: `~/Desktop/Photos2023`
   - Click "Export"

**On Windows (iCloud Photos):**
1. Open iCloud Photos in web browser
2. Filter by year 2023
3. Select photos (Ctrl + A for all)
4. Click download icon
5. Extract downloaded ZIP to folder

#### Step 2: Curate with Remember Twelve

```bash
cd ~/amplifier/remember-twelve
./sync_photos.sh ~/Desktop/Photos2023 2023
```

### Method C: Photos Library Direct Access (Advanced) ⚠️

For advanced users who understand the risks:

```bash
# Point directly to Photos library
./sync_photos.sh ~/Pictures/Photos\ Library.photoslibrary/originals 2023
```

**⚠️ Warnings:**
- Read-only access recommended
- Apple may change library structure
- May not preserve all metadata
- Export (Method A/B) is safer and recommended

## Detailed Workflows

### Exporting Specific Albums

**Method A (Automated - macOS):**
```bash
./sync_photos.sh --apple-photos 2023 --album "Family Vacation"
```

**Method B (Manual):**
1. Open Photos app
2. Click album in sidebar (e.g., "Family Vacation")
3. Select all photos in album (`Cmd + A`)
4. File → Export → Export Unmodified Originals
5. Save to folder
6. Run: `./sync_photos.sh <folder-path> 2023`

### Exporting Multiple Years

```bash
# Export each year separately
./sync_photos.sh --apple-photos 2022
./sync_photos.sh --apple-photos 2023
./sync_photos.sh --apple-photos 2024

# Or combine in one folder
# 1. Export 2022, 2023, 2024 to ~/Desktop/Photos2022-2024
# 2. Run: ./sync_photos.sh ~/Desktop/Photos2022-2024 2023
```

### Filtering by Favorites or People

**Export only Favorites:**
1. Photos app → Albums → Favorites
2. Select all → Export
3. Curate: `./sync_photos.sh <folder> 2023`

**Export photos with specific person:**
1. Photos app → Albums → People
2. Select person (e.g., "Sarah")
3. Filter by year in search: `date:2023`
4. Select all → Export
5. Curate: `./sync_photos.sh <folder> 2023`

## Export Options Explained

When exporting from Photos app, you'll see options:

### Export Type
- **✅ Export Unmodified Originals** (Recommended)
  - Original files with full quality
  - All EXIF metadata preserved
  - Best for Remember Twelve curation

- ❌ Export Photos
  - May apply edits
  - May reduce quality
  - Metadata might be lost

### Subfolder Format
- **None** - All photos in one folder (simplest)
- Moment Name - Organized by events
- Date - Organized by date folders

**Recommendation**: Choose "None" for simplest workflow.

### File Naming
- **Use File Name** - Keep original names
- Use Title - Use photo titles (if set)

**Recommendation**: "Use File Name" works best.

### Include Metadata
- **✅ Always check this box**
- Preserves GPS, dates, camera info
- Essential for quality curation

## Understanding Apple Photos Library Structure

If you're curious about the library structure:

```
~/Pictures/Photos Library.photoslibrary/
├── database/           # SQLite databases (private)
│   └── photos.db      # Photo metadata
├── originals/         # Original photo files
│   └── [hash folders]/
├── resources/         # Thumbnails, edited versions
└── scopes/           # iCloud sync data
```

**Why we don't access this directly:**
- Structure is Apple's private implementation
- Changes between macOS versions
- Requires parsing SQLite databases
- Export is cleaner and safer

## Comparing Methods

| Feature | Automated (macOS) | Manual Export | Direct Library |
|---------|-------------------|---------------|----------------|
| Platform | macOS only | All platforms | macOS only |
| Setup | One command | 2 min manual | Advanced setup |
| Speed | Fast | Fast | Fastest |
| Safety | Safe | Safe | Risky |
| Metadata | ✅ Full | ✅ Full | ⚠️ Partial |
| Recommended | ✅ Yes | ✅ Yes | ❌ No |

## Tips & Best Practices

### Storage Considerations

Export size depends on your library:
- **Small** (< 1000 photos/year): ~5-10 GB
- **Medium** (1000-5000 photos/year): ~20-50 GB
- **Large** (5000+ photos/year): ~100+ GB

**Tip**: Export to external drive if internal storage is limited.

### Keeping Exports Organized

```bash
# Create organized export structure
mkdir -p ~/RememberTwelve/exports
mkdir -p ~/RememberTwelve/exports/2023
mkdir -p ~/RememberTwelve/exports/2024

# Export to organized folders
# Photos app → Export to ~/RememberTwelve/exports/2023
```

### Re-using Exports

Once exported, you can re-run curation with different strategies:

```bash
# Try different curation strategies
./sync_photos.sh ~/RememberTwelve/exports/2023 2023 balanced
./sync_photos.sh ~/RememberTwelve/exports/2023 2023 aesthetic_first
./sync_photos.sh ~/RememberTwelve/exports/2023 2023 people_first
```

No need to re-export!

### Cleaning Up After Curation

```bash
# After successful curation, you can delete export
rm -rf ~/Desktop/Photos2023

# Or keep for future re-curation with different settings
```

## Troubleshooting

### "Photos app not found"

**On macOS**: Photos.app should be in `/Applications/Photos.app`
- If missing, reinstall from App Store

**On Windows**: Use manual export method via iCloud Photos web interface

### "Permission denied" when accessing Photos

1. System Preferences → Security & Privacy → Privacy
2. Select "Photos" in left sidebar
3. Ensure Remember Twelve / Terminal has access
4. Try running script again

### "No photos found for year 2023"

1. Check export folder contains photos:
   ```bash
   ls ~/Desktop/Photos2023
   ```
2. Verify photos have correct dates in EXIF:
   ```bash
   mdls ~/Desktop/Photos2023/IMG_1234.jpg | grep kMDItemContentCreationDate
   ```
3. Try flexible month mode:
   ```bash
   ./sync_photos.sh ~/Desktop/Photos2023 2023 --flexible-months
   ```

### Photos have wrong dates

Apple Photos sometimes adjusts dates. To see original date:
1. Photos app → Select photo
2. Window → Info (or Cmd + I)
3. Check "Date" field

If dates are wrong in Photos, they'll be wrong in export. Fix in Photos first.

### Export is very slow

Large libraries take time to export:
- 1000 photos: ~5 minutes
- 5000 photos: ~20 minutes
- 10000+ photos: ~1 hour

**Tip**: Export in background while doing other work.

## Advanced: AppleScript Automation

For power users, you can automate exports with AppleScript:

```applescript
tell application "Photos"
    set exportFolder to POSIX file "/Users/you/Desktop/Export2023"

    -- Export all photos from 2023
    set thePhotos to every media item whose date is greater than date "1/1/2023" and date is less than date "1/1/2024"

    export thePhotos to exportFolder
end tell
```

Save as `export_photos.scpt` and run:
```bash
osascript export_photos.scpt
```

This is what the automated method does internally!

## Comparison: Apple Photos vs Google Takeout

| Aspect | Apple Photos Export | Google Takeout |
|--------|---------------------|----------------|
| Speed | ✅ Immediate (minutes) | ❌ Slow (hours to days) |
| Process | ✅ Direct from app | ❌ Web form, wait, download |
| Platforms | macOS, iOS, Windows (iCloud) | All platforms |
| Metadata | ✅ Excellent | ✅ Excellent |
| Ease | ✅ Very easy | ⚠️ Multiple steps |
| Ongoing | Can automate | Must repeat |

**Verdict**: For Apple users, Apple Photos is significantly easier than Google Takeout.

## FAQ

### Do I need iCloud Photos enabled?

**No.** You can export from local Photos library whether or not iCloud is enabled.

**But**: If you have iCloud Photos, your full library is accessible on any device.

### Will this work with iCloud Photos on Windows?

**Yes**, via manual export:
1. Go to iCloud.com → Photos
2. Download photos for your year
3. Point Remember Twelve to downloaded folder

### Can I export Live Photos?

**Yes.** Choose "Export Unmodified Originals" to get both the photo and video components.

Remember Twelve will curate based on the still image.

### What about edited photos?

"Export Unmodified Originals" exports the original, unedited version.

If you want edited versions:
- Use "Export Photos" option (may lose metadata)
- Or apply edits as originals (Photos → Image → Revert to Original first, then export)

### Can I export from iPhone/iPad?

**Yes**, but requires extra steps:
1. Select photos in Photos app
2. Tap Share icon
3. Save to Files app
4. Transfer to computer
5. Point Remember Twelve to folder

Or use iCloud Photos sync to access on Mac.

## Next Steps

After exporting and curating:

1. ✅ View your Remember Twelve in web UI
2. ✅ Try different curation strategies
3. ✅ Export to different formats
4. ✅ Set up annual reminder to export next year's photos

Remember: The export is a **one-time annual task**, but you can re-curate with different strategies anytime!

---

**Questions?** See [GOOGLE_PHOTOS_DEPRECATION.md](../GOOGLE_PHOTOS_DEPRECATION.md) for why we use exports instead of APIs.
