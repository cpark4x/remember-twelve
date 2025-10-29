from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PhotoResponse(BaseModel):
    id: int
    filename: str
    captured_at: str
    month: int
    year: int
    quality_score: Optional[float] = None
    emotional_score: Optional[float] = None
    combined_score: Optional[float] = None
    source_path: Optional[str] = None
    metadata_json: Optional[str] = None
    month_slot: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": 42,
                "filename": "IMG_2024_05_15.jpg",
                "captured_at": "2024-05-15T14:30:00",
                "month": 5,
                "year": 2024,
                "quality_score": 0.85,
                "emotional_score": 0.92,
                "combined_score": 0.88,
                "source_path": "/photos/2024/IMG_2024_05_15.jpg",
                "metadata_json": '{"camera": "iPhone 13"}',
                "month_slot": 5
            }
        }


class YearResponse(BaseModel):
    year: int
    photo_count: int
    has_curation: bool

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2024,
                "photo_count": 487,
                "has_curation": True
            }
        }


class TwelveResponse(BaseModel):
    year: int
    strategy: str
    curation_id: int
    photos: List[PhotoResponse] = Field(description="Exactly 12 photos, one per month")

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2024,
                "strategy": "balanced",
                "curation_id": 1,
                "photos": []
            }
        }


class AlternativesResponse(BaseModel):
    year: int
    month: int
    current_photo: PhotoResponse
    alternatives: List[PhotoResponse]

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2024,
                "month": 5,
                "current_photo": {},
                "alternatives": []
            }
        }


class SwapRequest(BaseModel):
    year: int
    month_slot: int
    new_photo_id: int
    reason: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2024,
                "month_slot": 5,
                "new_photo_id": 128,
                "reason": "Better composition"
            }
        }


class SwapResponse(BaseModel):
    success: bool
    message: str
    swap_id: int

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Photo swapped successfully",
                "swap_id": 42
            }
        }


class SyncRequest(BaseModel):
    year: int
    force_refresh: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2024,
                "force_refresh": False
            }
        }


class SyncResponse(BaseModel):
    success: bool
    message: str
    photos_synced: int
    year: int

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Synced 487 photos for 2024",
                "photos_synced": 487,
                "year": 2024
            }
        }


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    database_connected: bool

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "database_connected": True
            }
        }


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Year 2024 not found",
                "error_code": "YEAR_NOT_FOUND"
            }
        }


class CurateRequest(BaseModel):
    year: int = Field(..., ge=1900, le=2100, description="Year to curate")
    strategy: str = Field(default="balanced", description="Curation strategy")
    force_refresh: bool = Field(default=False, description="Force re-curation even if one exists")

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2024,
                "strategy": "balanced",
                "force_refresh": False
            }
        }


class CurationStats(BaseModel):
    photos_analyzed: int
    photos_selected: int
    strategy_used: str
    duration_seconds: float

    class Config:
        json_schema_extra = {
            "example": {
                "photos_analyzed": 487,
                "photos_selected": 12,
                "strategy_used": "balanced",
                "duration_seconds": 2.3
            }
        }


class CurateResponse(BaseModel):
    curation_id: int
    year: int
    strategy: str
    photos_analyzed: int
    photos_selected: int
    stats: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "curation_id": 1,
                "year": 2024,
                "strategy": "balanced",
                "photos_analyzed": 487,
                "photos_selected": 12,
                "stats": {
                    "photos_analyzed": 487,
                    "photos_selected": 12,
                    "strategy_used": "balanced",
                    "duration_seconds": 2.3
                }
            }
        }
