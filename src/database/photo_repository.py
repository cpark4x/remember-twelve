import sqlite3
from typing import List, Optional


def insert_photo(conn: sqlite3.Connection, photo_data: dict) -> int:
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO photos (
            filename, source_path, captured_at, month, year,
            quality_score, emotional_score, combined_score, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            photo_data["filename"],
            photo_data.get("source_path"),
            photo_data["captured_at"],
            photo_data["month"],
            photo_data["year"],
            photo_data.get("quality_score"),
            photo_data.get("emotional_score"),
            photo_data.get("combined_score"),
            photo_data.get("metadata_json")
        )
    )

    return cursor.lastrowid


def insert_photos_batch(conn: sqlite3.Connection, photos: List[dict]) -> List[int]:
    photo_ids = []
    for photo_data in photos:
        photo_id = insert_photo(conn, photo_data)
        photo_ids.append(photo_id)
    return photo_ids


def get_photo_by_id(conn: sqlite3.Connection, photo_id: int) -> Optional[dict]:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def get_photo_by_filename(conn: sqlite3.Connection, filename: str, year: int) -> Optional[dict]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM photos WHERE filename = ? AND year = ?",
        (filename, year)
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def get_photos_by_year(conn: sqlite3.Connection, year: int) -> List[dict]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM photos WHERE year = ? ORDER BY captured_at",
        (year,)
    )
    return [dict(row) for row in cursor.fetchall()]


def get_photos_by_month(conn: sqlite3.Connection, year: int, month: int) -> List[dict]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM photos
        WHERE year = ? AND month = ?
        ORDER BY combined_score DESC, captured_at
        """,
        (year, month)
    )
    return [dict(row) for row in cursor.fetchall()]


def get_top_photos(conn: sqlite3.Connection, year: int, limit: int = 12) -> List[dict]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM photos
        WHERE year = ?
        ORDER BY combined_score DESC, captured_at
        LIMIT ?
        """,
        (year, limit)
    )
    return [dict(row) for row in cursor.fetchall()]


def get_top_photos_by_month(
    conn: sqlite3.Connection,
    year: int,
    month: int,
    limit: int = 1
) -> List[dict]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM photos
        WHERE year = ? AND month = ?
        ORDER BY combined_score DESC, captured_at
        LIMIT ?
        """,
        (year, month, limit)
    )
    return [dict(row) for row in cursor.fetchall()]


def get_month_distribution(conn: sqlite3.Connection, year: int) -> dict:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT month, COUNT(*) as count
        FROM photos
        WHERE year = ?
        GROUP BY month
        ORDER BY month
        """,
        (year,)
    )

    distribution = {month: 0 for month in range(1, 13)}
    for row in cursor.fetchall():
        distribution[row["month"]] = row["count"]

    return distribution


def get_score_stats(conn: sqlite3.Connection, year: int) -> dict:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            COUNT(*) as count,
            AVG(quality_score) as avg_quality,
            AVG(emotional_score) as avg_emotional,
            AVG(combined_score) as avg_combined,
            MIN(combined_score) as min_score,
            MAX(combined_score) as max_score
        FROM photos
        WHERE year = ?
        """,
        (year,)
    )

    row = cursor.fetchone()
    return dict(row) if row else {}


def update_photo_scores(
    conn: sqlite3.Connection,
    photo_id: int,
    quality_score: Optional[float] = None,
    emotional_score: Optional[float] = None,
    combined_score: Optional[float] = None
) -> None:
    updates = []
    params = []

    if quality_score is not None:
        updates.append("quality_score = ?")
        params.append(quality_score)

    if emotional_score is not None:
        updates.append("emotional_score = ?")
        params.append(emotional_score)

    if combined_score is not None:
        updates.append("combined_score = ?")
        params.append(combined_score)

    if not updates:
        return

    params.append(photo_id)

    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE photos SET {', '.join(updates)} WHERE id = ?",
        tuple(params)
    )


def delete_photo(conn: sqlite3.Connection, photo_id: int) -> None:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM photos WHERE id = ?", (photo_id,))


def delete_photos_by_year(conn: sqlite3.Connection, year: int) -> int:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM photos WHERE year = ?", (year,))
    return cursor.rowcount


def get_years(conn: sqlite3.Connection) -> List[int]:
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT year FROM photos ORDER BY year DESC")
    return [row["year"] for row in cursor.fetchall()]
