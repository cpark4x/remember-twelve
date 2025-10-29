#!/usr/bin/env python3
"""
Migrate existing JSON curation data to SQLite database.

Usage:
    python migrate_to_database.py [json_file]

If no json_file specified, tries:
    1. ui/twelve_2023_balanced.json
    2. twelve_2023_balanced.json
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from src.database import init_db, get_connection
from src.database.photo_repository import insert_photo, get_photo_by_filename
from src.database.curation_repository import (
    create_curation,
    add_curation_photos_batch,
    get_active_curation
)


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def load_json_data(json_path: Optional[str] = None) -> dict:
    """Load curation JSON file"""
    if json_path:
        paths = [Path(json_path)]
    else:
        paths = [
            Path("ui/twelve_2023_balanced.json"),
            Path("twelve_2023_balanced.json")
        ]

    for path in paths:
        if path.exists():
            print(f"📂 Loading: {path}")
            with open(path) as f:
                return json.load(f)

    raise FileNotFoundError(
        f"No JSON file found. Tried: {', '.join(str(p) for p in paths)}"
    )


def parse_timestamp(timestamp_str: str) -> str:
    """Parse timestamp string to ISO format"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.isoformat()
    except ValueError:
        return timestamp_str


def import_photos(conn, photos_data: list, year: int) -> Dict[str, int]:
    """
    Import photos, return mapping of original_path -> db_id
    Skips duplicates based on filename + captured_at
    """
    photo_map = {}
    imported = 0
    skipped = 0

    for photo in photos_data:
        source_path = photo["photo_path"]
        filename = Path(source_path).name

        captured_at = parse_timestamp(photo["timestamp"])
        month = photo["month"]

        existing = get_photo_by_filename(conn, filename, year)
        if existing:
            photo_map[source_path] = existing["id"]
            skipped += 1
            continue

        metadata_json = json.dumps(photo.get("metadata", {}))

        photo_data = {
            "filename": filename,
            "source_path": source_path,
            "captured_at": captured_at,
            "month": month,
            "year": year,
            "quality_score": photo.get("quality_score"),
            "emotional_score": photo.get("emotional_score"),
            "combined_score": photo.get("combined_score"),
            "metadata_json": metadata_json
        }

        photo_id = insert_photo(conn, photo_data)
        photo_map[source_path] = photo_id
        imported += 1

    print(f"  ✓ Imported {imported} photos (skipped {skipped} duplicates)")
    return photo_map


def create_curation_record(
    conn,
    year: int,
    strategy: str,
    stats: dict
) -> int:
    """Create curation record"""
    curation_id = create_curation(conn, year, strategy, stats)
    return curation_id


def assign_photos_to_months(
    conn,
    curation_id: int,
    month_dist: dict,
    photo_map: Dict[str, int]
):
    """Link photos to curation with month assignments"""
    assignments = []

    for month_name in MONTH_NAMES:
        if month_name not in month_dist:
            print(f"  ⚠️  Warning: No photo for {month_name}")
            continue

        photo_data = month_dist[month_name]
        photo_path = photo_data["photo_path"]

        if photo_path not in photo_map:
            print(f"  ⚠️  Warning: Photo not found for {month_name}: {photo_path}")
            continue

        photo_id = photo_map[photo_path]
        month_slot = MONTH_NAMES.index(month_name) + 1

        assignments.append((photo_id, month_slot))

    if assignments:
        add_curation_photos_batch(conn, curation_id, assignments)

    print(f"  ✓ Assigned {len(assignments)} photos to months")


def verify_migration(conn, year: int):
    """Verify migration was successful"""
    from src.database.curation_repository import get_twelve_photos

    twelve = get_twelve_photos(conn, year)

    if len(twelve) != 12:
        print(f"  ⚠️  Warning: Expected 12 photos, found {len(twelve)}")
        return False

    for photo in twelve:
        month_slot = photo.get("month_slot")
        if month_slot < 1 or month_slot > 12:
            print(f"  ⚠️  Warning: Invalid month_slot {month_slot} for photo {photo['id']}")
            return False

    print(f"  ✓ Verification passed: 12 photos correctly assigned")
    return True


def main():
    print("=" * 60)
    print("🗄️  Remember Twelve - Data Migration")
    print("=" * 60)
    print()

    json_file = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        data = load_json_data(json_file)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    year = data.get("year", 2023)
    strategy = data.get("strategy", "balanced")
    stats = data.get("stats", {})
    photos = data.get("photos", [])
    month_dist = data.get("month_distribution", {})

    print(f"📅 Year: {year}")
    print(f"🎯 Strategy: {strategy}")
    print(f"📸 Photos: {len(photos)}")
    print(f"📊 Stats: {stats}")
    print()

    init_db()
    print("✓ Database initialized")
    print()

    with get_connection() as conn:
        existing_curation = get_active_curation(conn, year)
        if existing_curation:
            print(f"⚠️  Warning: Active curation already exists for {year}")
            response = input("   Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                sys.exit(0)
            print()

        print("📥 Importing photos...")
        photo_map = import_photos(conn, photos, year)
        print()

        print("📝 Creating curation...")
        curation_id = create_curation_record(conn, year, strategy, stats)
        print(f"  ✓ Created curation #{curation_id}")
        print()

        print("🗓️  Assigning photos to months...")
        assign_photos_to_months(conn, curation_id, month_dist, photo_map)
        print()

        print("✅ Verifying migration...")
        verify_migration(conn, year)
        print()

        conn.commit()

    print("=" * 60)
    print("✅ Migration complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  • View your twelve: python src/twelve_curator/viewer.py")
    print("  • Run curation: python curate_local_photos.py")
    print()


if __name__ == "__main__":
    main()
