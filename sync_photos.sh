#!/bin/bash
# Curate photos from local folder (Google Takeout, local photos, etc.)

set -e  # Exit on error

# Parse arguments
SOURCE_PATH="$1"
YEAR="${2:-2023}"
STRATEGY="${3:-balanced}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$SCRIPT_DIR/ui"

# Show usage if no source path provided
if [ -z "$SOURCE_PATH" ]; then
    echo "Usage: $0 <source_path> [year] [strategy]"
    echo ""
    echo "Examples:"
    echo "  $0 ~/Downloads/Takeout/Google\ Photos 2023"
    echo "  $0 ~/Pictures/2023 2023 aesthetic_first"
    echo "  $0 ~/GoogleDrive/Google\ Photos 2024 balanced"
    echo ""
    echo "Available strategies:"
    echo "  balanced (default) - Even quality + emotional significance"
    echo "  aesthetic_first - Prioritize visual quality"
    echo "  people_first - Prioritize faces and emotions"
    echo "  top_heavy - More photos from best months"
    echo ""
    echo "See docs/GOOGLE_TAKEOUT_GUIDE.md for Google Takeout workflow"
    exit 1
fi

echo "📂 Source: $SOURCE_PATH"
echo "📅 Year: $YEAR"
echo "🎯 Strategy: $STRATEGY"
echo ""
echo "🔄 Curating photos..."

# Run curation with flexible month distribution
python3 "$SCRIPT_DIR/curate_from_google_photos.py" \
    --source "$SOURCE_PATH" \
    --year "$YEAR" \
    --strategy "$STRATEGY" \
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
