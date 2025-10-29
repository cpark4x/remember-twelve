import sqlite3
import logging
import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_connection
from ..models import TwelveResponse, AlternativesResponse, PhotoResponse, CurateRequest, CurateResponse, ErrorResponse
from src.database.curation_repository import get_active_curation, get_twelve_photos
from src.database.photo_repository import get_photos_by_month, get_photos_by_year
from src.services.curation_service import CurationService, AuthenticationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/year", tags=["curations"])


@router.get("/{year}/twelve", response_model=TwelveResponse)
def get_twelve(year: int, conn: sqlite3.Connection = Depends(get_connection)):
    curation = get_active_curation(conn, year)

    if not curation:
        raise HTTPException(status_code=404, detail=f"No active curation found for year {year}")

    photos = get_twelve_photos(conn, year)

    if len(photos) != 12:
        raise HTTPException(
            status_code=500,
            detail=f"Expected 12 photos, found {len(photos)}"
        )

    photo_responses = [PhotoResponse(**photo) for photo in photos]

    return TwelveResponse(
        year=year,
        strategy=curation["strategy"],
        curation_id=curation["id"],
        photos=photo_responses
    )


@router.get("/{year}/alternatives/{month}", response_model=AlternativesResponse)
def get_alternatives(
    year: int,
    month: int,
    conn: sqlite3.Connection = Depends(get_connection)
):
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")

    curation = get_active_curation(conn, year)

    if not curation:
        raise HTTPException(status_code=404, detail=f"No active curation found for year {year}")

    twelve_photos = get_twelve_photos(conn, year)
    current_photo = next((p for p in twelve_photos if p["month_slot"] == month), None)

    if not current_photo:
        raise HTTPException(status_code=404, detail=f"No photo found for month {month}")

    all_month_photos = get_photos_by_month(conn, year, month)

    alternatives = [p for p in all_month_photos if p["id"] != current_photo["id"]]

    return AlternativesResponse(
        year=year,
        month=month,
        current_photo=PhotoResponse(**current_photo),
        alternatives=[PhotoResponse(**p) for p in alternatives]
    )


@router.get("/{year}/photos", response_model=List[PhotoResponse])
def get_all_photos(year: int, conn: sqlite3.Connection = Depends(get_connection)):
    """Get all photos for a year, sorted by combined score descending"""
    photos = get_photos_by_year(conn, year)

    if not photos:
        raise HTTPException(status_code=404, detail=f"No photos found for year {year}")

    # Sort by combined score descending
    sorted_photos = sorted(photos, key=lambda p: p.get('combined_score', 0), reverse=True)

    return [PhotoResponse(**photo) for photo in sorted_photos]


# Create a separate router for curate endpoint (not under /api/year prefix)
curate_router = APIRouter(prefix="/api", tags=["curation"])


@curate_router.post("/curate", response_model=CurateResponse, responses={
    401: {"model": ErrorResponse, "description": "Authentication required"},
    404: {"model": ErrorResponse, "description": "No photos found for year"},
    500: {"model": ErrorResponse, "description": "Curation pipeline failed"}
})
def curate_year(
    request: CurateRequest,
    conn: sqlite3.Connection = Depends(get_connection)
) -> CurateResponse:
    """
    Run curation pipeline for a specific year.

    Analyzes all photos for the given year and selects 12 photos (one per month)
    using the specified strategy. Returns the curation ID and statistics about
    the curation process.

    - **year**: Year to curate (1900-2100)
    - **strategy**: Curation strategy (default: "balanced")
    - **force_refresh**: Re-curate even if curation exists (default: False)
    """
    start_time = time.time()

    try:
        # Initialize service
        service = CurationService()

        # Get credentials path (default location)
        credentials_path = "google_photos_credentials.json"

        # Run curation pipeline
        logger.info(f"Starting curation for year {request.year} with strategy {request.strategy}")

        selection = service.curate_year(
            year=request.year,
            strategy=request.strategy,
            credentials_path=credentials_path,
            progress_callback=None  # Synchronous - no progress updates for now
        )

        # Save to database in a transaction
        try:
            conn.execute("BEGIN")
            curation_id = service.save_to_database(conn, selection)
            conn.commit()
            logger.info(f"Curation {curation_id} saved successfully for year {request.year}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error during curation save: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save curation to database: {str(e)}"
            )

        # Calculate stats
        duration = time.time() - start_time
        photos_analyzed = len(selection.all_candidates) if hasattr(selection, 'all_candidates') else 0
        photos_selected = len(selection.photos)

        return CurateResponse(
            curation_id=curation_id,
            year=request.year,
            strategy=request.strategy,
            photos_analyzed=photos_analyzed,
            photos_selected=photos_selected,
            stats={
                "photos_analyzed": photos_analyzed,
                "photos_selected": photos_selected,
                "strategy_used": request.strategy,
                "duration_seconds": round(duration, 2)
            }
        )

    except AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Google Photos authentication required: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Curation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Curation pipeline failed: {str(e)}"
        )
