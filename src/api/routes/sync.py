import sqlite3
import json
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_connection, get_photos_dir
from ..models import SyncRequest, SyncResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sync", tags=["sync"])


@router.post("", response_model=SyncResponse)
def sync_google_photos(
    sync_request: SyncRequest,
    conn: sqlite3.Connection = Depends(get_connection)
):
    try:
        from src.google_photos import GooglePhotosClient
        from src.database.photo_repository import (
            insert_photo,
            get_photo_by_filename,
            delete_photos_by_year
        )

        client = GooglePhotosClient()

        if not client.is_authenticated():
            raise HTTPException(
                status_code=401,
                detail="Not authenticated with Google Photos. Please run authentication first."
            )

        if sync_request.force_refresh:
            deleted_count = delete_photos_by_year(conn, sync_request.year)
            logger.info(f"Deleted {deleted_count} existing photos for year {sync_request.year}")

        photos = client.get_photos_for_year(sync_request.year)

        if not photos:
            return SyncResponse(
                success=True,
                message=f"No photos found for year {sync_request.year}",
                photos_synced=0,
                year=sync_request.year
            )

        photos_dir = get_photos_dir()
        year_dir = photos_dir / str(sync_request.year)
        year_dir.mkdir(parents=True, exist_ok=True)

        synced_count = 0
        skipped_count = 0

        for idx, photo_item in enumerate(photos):
            filename = photo_item.get("filename", f"photo_{idx}.jpg")

            existing = get_photo_by_filename(conn, filename, sync_request.year)
            if existing and not sync_request.force_refresh:
                skipped_count += 1
                continue

            local_path = year_dir / filename
            base_url = photo_item.get("baseUrl")

            if base_url:
                success = client.download_photo(base_url, str(local_path))
                if not success:
                    logger.warning(f"Failed to download: {filename}")
                    continue

            creation_time = photo_item.get("mediaMetadata", {}).get("creationTime")
            if creation_time:
                from datetime import datetime
                dt = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
                month = dt.month
                captured_at = dt.isoformat()
            else:
                month = 1
                captured_at = f"{sync_request.year}-01-01T00:00:00"

            photo_data = {
                "filename": filename,
                "source_path": str(local_path.relative_to(photos_dir.parent)),
                "captured_at": captured_at,
                "month": month,
                "year": sync_request.year,
                "quality_score": None,
                "emotional_score": None,
                "combined_score": None,
                "metadata_json": json.dumps(photo_item.get("mediaMetadata", {}))
            }

            insert_photo(conn, photo_data)
            synced_count += 1

        return SyncResponse(
            success=True,
            message=f"Synced {synced_count} photos for {sync_request.year} (skipped {skipped_count} existing)",
            photos_synced=synced_count,
            year=sync_request.year
        )

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Google Photos integration not available"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sync failed: {str(e)}"
        )
