#!/usr/bin/env python3
"""
Export curated photos to a viewable folder.
Copies all curated photos from the JSON file to a single directory.
"""

import json
import shutil
from pathlib import Path

def export_curated_photos(json_file: str, output_dir: str):
    """Export curated photos to a directory."""

    # Load the curation data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📸 Remember Twelve - Exporting Curated Photos")
    print(f"=" * 60)
    print(f"Year: {data['year']}")
    print(f"Strategy: {data['strategy']}")
    print(f"Photos to export: {len(data['photos'])}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"=" * 60)
    print()

    # Copy each photo
    copied = 0
    for i, photo in enumerate(data['photos'], 1):
        src = Path(photo['photo_path'])

        if not src.exists():
            print(f"❌ {i}. Photo not found: {src.name}")
            continue

        # Create a numbered filename with rank
        ext = src.suffix
        dest_name = f"{i:02d}_score-{int(photo['combined_score'])}_{src.name}"
        dest = output_path / dest_name

        # Copy the file
        try:
            shutil.copy2(src, dest)
            score_emoji = "✅" if photo['combined_score'] >= 70 else "⚠️" if photo['combined_score'] >= 50 else "📷"
            print(f"{score_emoji} {i}. Copied: {dest.name}")
            copied += 1
        except Exception as e:
            print(f"❌ {i}. Error copying {src.name}: {e}")

    print()
    print(f"=" * 60)
    print(f"✅ Exported {copied}/{len(data['photos'])} photos")
    print(f"📂 Open folder: open {output_path.absolute()}")
    print(f"=" * 60)

    return output_path

if __name__ == '__main__':
    import sys

    # Default values
    json_file = 'twelve_2023_balanced.json'
    output_dir = 'curated_photos_2023'

    # Allow custom output directory from command line
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    output_path = export_curated_photos(json_file, output_dir)

    # Automatically open the folder in Finder
    import subprocess
    subprocess.run(['open', str(output_path)])
