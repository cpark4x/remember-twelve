import sqlite3
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_connection
from ..models import YearResponse
from src.database.photo_repository import get_years
from src.database.curation_repository import get_active_curation

router = APIRouter(prefix="/api/years", tags=["years"])


@router.get("", response_model=List[YearResponse])
def list_years(conn: sqlite3.Connection = Depends(get_connection)):
    years = get_years(conn)

    if not years:
        return []

    results = []
    for year in years:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM photos WHERE year = ?", (year,))
        photo_count = cursor.fetchone()["count"]

        curation = get_active_curation(conn, year)
        has_curation = curation is not None

        results.append(YearResponse(
            year=year,
            photo_count=photo_count,
            has_curation=has_curation
        ))

    return results
