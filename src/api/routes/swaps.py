import sqlite3
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_connection
from ..models import SwapRequest, SwapResponse
from src.database.curation_repository import get_active_curation, execute_swap

router = APIRouter(prefix="/api/swaps", tags=["swaps"])


@router.post("", response_model=SwapResponse)
def save_photo_swap(
    swap: SwapRequest,
    conn: sqlite3.Connection = Depends(get_connection)
):
    curation = get_active_curation(conn, swap.year)

    if not curation:
        raise HTTPException(
            status_code=404,
            detail=f"No active curation found for year {swap.year}"
        )

    if swap.month_slot < 1 or swap.month_slot > 12:
        raise HTTPException(
            status_code=400,
            detail="Month slot must be between 1 and 12"
        )

    cursor = conn.cursor()
    cursor.execute("SELECT id FROM photos WHERE id = ?", (swap.new_photo_id,))
    photo_exists = cursor.fetchone()

    if not photo_exists:
        raise HTTPException(
            status_code=404,
            detail=f"Photo {swap.new_photo_id} not found"
        )

    try:
        execute_swap(
            conn,
            curation["id"],
            swap.month_slot,
            swap.new_photo_id,
            swap.reason
        )

        cursor.execute("SELECT last_insert_rowid() as id")
        swap_id = cursor.fetchone()["id"]

        return SwapResponse(
            success=True,
            message=f"Successfully swapped photo for month {swap.month_slot}",
            swap_id=swap_id
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save swap: {str(e)}")
