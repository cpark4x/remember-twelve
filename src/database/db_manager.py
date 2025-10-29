import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional


DEFAULT_DB_PATH = os.path.expanduser("~/.remember_twelve/remember_twelve.db")
SCHEMA_VERSION = 1


def _get_schema_path() -> Path:
    return Path(__file__).parent / "schema.sql"


def init_db(db_path: Optional[str] = None) -> None:
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    db_path = os.path.expanduser(db_path)
    db_dir = os.path.dirname(db_path)

    os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        schema_file = _get_schema_path()
        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        conn.executescript(schema_sql)

        run_migrations(conn)

        conn.commit()
    finally:
        conn.close()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    db_path = os.path.expanduser(db_path)

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found at {db_path}. Call init_db() first."
        )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


@contextmanager
def transaction(db_path: Optional[str] = None):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_migrations(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(version) FROM schema_version")
    result = cursor.fetchone()
    current_version = result[0] if result[0] is not None else 0

    if current_version < SCHEMA_VERSION:
        cursor.execute(
            "INSERT INTO schema_version (version, description) VALUES (?, ?)",
            (SCHEMA_VERSION, "Initial schema")
        )


def get_db_stats(db_path: Optional[str] = None) -> dict:
    with transaction(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM photos")
        photo_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT year) FROM photos")
        year_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM curations WHERE is_active = 1")
        active_curation_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM swaps")
        swap_count = cursor.fetchone()[0]

        cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        result = cursor.fetchone()
        schema_version = result[0] if result else 0

        return {
            "photo_count": photo_count,
            "year_count": year_count,
            "active_curations": active_curation_count,
            "swap_count": swap_count,
            "schema_version": schema_version
        }
