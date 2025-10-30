#!/bin/bash
# Curate photos from local folder, Google Takeout, or Apple Photos

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$SCRIPT_DIR/ui"

# Parse arguments
APPLE_PHOTOS=false
MANUAL=false
ALBUM=""
SOURCE_PATH=""
YEAR="2023"
STRATEGY="balanced"

# Parse flags and arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --apple-photos)
            APPLE_PHOTOS=true
            shift
            ;;
        --manual)
            MANUAL=true
            shift
            ;;
        --album)
            ALBUM="$2"
            shift 2
            ;;
        *)
            if [ -z "$SOURCE_PATH" ]; then
                SOURCE_PATH="$1"
            elif [ -z "$YEAR" ] || [ "$YEAR" == "2023" ]; then
                YEAR="$1"
            else
                STRATEGY="$1"
            fi
            shift
            ;;
    esac
done

# Show usage if no source path and not Apple Photos mode
if [ -z "$SOURCE_PATH" ] && [ "$APPLE_PHOTOS" = false ]; then
    echo "Usage: $0 <source_path> [year] [strategy]"
    echo "   OR: $0 --apple-photos [year] [--album <name>] [--manual]"
    echo ""
    echo "📁 Local Folder Examples:"
    echo "  $0 ~/Downloads/Takeout/Google\ Photos 2023"
    echo "  $0 ~/Pictures/2023 2023 aesthetic_first"
    echo "  $0 ~/GoogleDrive/Google\ Photos 2024 balanced"
    echo ""
    echo "🍎 Apple Photos Examples:"
    echo "  $0 --apple-photos 2023                    # Automated export (macOS)"
    echo "  $0 --apple-photos 2023 --album Favorites  # Export specific album"
    echo "  $0 --apple-photos 2023 --manual          # Show manual instructions"
    echo ""
    echo "Available strategies:"
    echo "  balanced (default) - Even quality + emotional significance"
    echo "  aesthetic_first - Prioritize visual quality"
    echo "  people_first - Prioritize faces and emotions"
    echo "  top_heavy - More photos from best months"
    echo ""
    echo "📖 Guides:"
    echo "  Google Takeout: docs/GOOGLE_TAKEOUT_GUIDE.md"
    echo "  Apple Photos:   docs/APPLE_PHOTOS_GUIDE.md"
    exit 1
fi

# Handle Apple Photos mode
if [ "$APPLE_PHOTOS" = true ]; then
    echo "🍎 Apple Photos Mode"
    echo "📅 Year: $YEAR"
    if [ -n "$ALBUM" ]; then
        echo "📁 Album: $ALBUM"
    fi
    echo "🎯 Strategy: $STRATEGY"
    echo ""

    # Build Python command for Apple Photos export
    PYTHON_CMD="from src.photo_sources.apple_photos_helper import export_from_apple_photos; "
    PYTHON_CMD+="path, automated = export_from_apple_photos("
    PYTHON_CMD+="year=$YEAR, "
    PYTHON_CMD+="album_name='$ALBUM' if '$ALBUM' else None, "
    PYTHON_CMD+="manual=$([[ "$MANUAL" == true ]] && echo True || echo False)"
    PYTHON_CMD+="); "
    PYTHON_CMD+="print(f'EXPORT_PATH={path}') if path else exit(1)"

    # Run export
    EXPORT_RESULT=$(python3 -c "$PYTHON_CMD" 2>&1)
    EXPORT_STATUS=$?

    if [ $EXPORT_STATUS -ne 0 ]; then
        echo ""
        echo "ℹ️  Follow the manual instructions above, then run:"
        echo "   $0 <exported-folder-path> $YEAR $STRATEGY"
        exit 0
    fi

    # Extract export path from result
    SOURCE_PATH=$(echo "$EXPORT_RESULT" | grep "EXPORT_PATH=" | cut -d= -f2)

    if [ -z "$SOURCE_PATH" ]; then
        echo "❌ Failed to get export path"
        exit 1
    fi

    echo ""
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
