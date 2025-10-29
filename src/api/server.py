import sqlite3
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from .routes import years_router, curations_router, curate_router, swaps_router, sync_router
from .models import HealthResponse
from .dependencies import get_connection, get_photos_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Remember Twelve API",
    description="Unified API for photo curation and management",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(years_router)
app.include_router(curations_router)
app.include_router(curate_router)
app.include_router(swaps_router)
app.include_router(sync_router)


@app.get("/", response_class=HTMLResponse)
def serve_viewer():
    ui_dir = Path(__file__).parent.parent.parent / "ui"
    viewer_path = ui_dir / "viewer_dynamic.html"

    if not viewer_path.exists():
        raise HTTPException(status_code=404, detail="Viewer not found")

    # Disable caching for development
    return FileResponse(
        viewer_path,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    try:
        conn_gen = get_connection()
        conn = next(conn_gen)
        conn.execute("SELECT 1")
        database_connected = True
        conn.close()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        database_connected = False

    return HealthResponse(
        status="healthy" if database_connected else "degraded",
        version="1.0.0",
        database_connected=database_connected
    )


@app.get("/photos/{year}/{filename}")
def serve_photo(year: int, filename: str):
    photos_dir = get_photos_dir()
    photo_path = photos_dir / str(year) / filename

    if not photo_path.exists():
        legacy_path = photos_dir / filename
        if legacy_path.exists():
            return FileResponse(legacy_path)
        raise HTTPException(status_code=404, detail="Photo not found")

    return FileResponse(photo_path)


ui_dir = Path(__file__).parent.parent.parent / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir)), name="ui")


@app.on_event("startup")
async def startup_event():
    logger.info("Remember Twelve API starting up...")
    logger.info(f"Photos directory: {get_photos_dir()}")
    logger.info(f"UI directory: {ui_dir}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Remember Twelve API shutting down...")
