import sqlite3
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_connection
from ..models import TwelveResponse, AlternativesResponse, PhotoResponse
from src.database.curation_repository import get_active_curation, get_twelve_photos
from src.database.photo_repository import get_photos_by_month, get_photos_by_year

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
