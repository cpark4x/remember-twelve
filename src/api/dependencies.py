import sqlite3
import os
from typing import Generator
from pathlib import Path


def get_db_path() -> Path:
    """Get the database path from user's home directory"""
    db_path = os.path.expanduser("~/.remember_twelve/remember_twelve.db")
    return Path(db_path)


def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get database connection with proper configuration"""
    db_path = get_db_path()

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run 'python remember_twelve_app.py start' to initialize."
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_photos_dir() -> Path:
    """Get the photos directory from user's home directory"""
    photos_path = os.path.expanduser("~/.remember_twelve/photos")
    return Path(photos_path)
