#!/bin/bash
# Sync photos from Google Photos and update viewer

set -e  # Exit on error

YEAR=${1:-2023}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$SCRIPT_DIR/ui"

echo "🔄 Syncing photos from Google Photos for year $YEAR..."

# Run curation with flexible month distribution
python3 "$SCRIPT_DIR/curate_from_google_photos.py" \
    --year "$YEAR" \
    --strategy balanced \
    --flexible-months \
    --output "$UI_DIR/photos_data.json"

# Copy photos to UI directory
if [ -f "$SCRIPT_DIR/twelve_${YEAR}_balanced.json" ]; then
    python3 "$SCRIPT_DIR/copy_photos_to_ui.py"
    echo "✅ Sync complete! Photos updated in $UI_DIR/photos/"
else
    echo "❌ Sync failed. No output file generated."
    exit 1
fi
