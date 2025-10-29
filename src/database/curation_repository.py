import sqlite3
from typing import List, Optional, Tuple


def create_curation(
    conn: sqlite3.Connection,
    year: int,
    strategy: str,
    stats: dict
) -> int:
    import json

    cursor = conn.cursor()

    cursor.execute(
        "UPDATE curations SET is_active = 0 WHERE year = ? AND is_active = 1",
        (year,)
    )

    cursor.execute(
        """
        INSERT INTO curations (year, strategy, stats_json, is_active)
        VALUES (?, ?, ?, 1)
        """,
        (year, strategy, json.dumps(stats))
    )

    return cursor.lastrowid


def get_curation_by_id(conn: sqlite3.Connection, curation_id: int) -> Optional[dict]:
    import json

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM curations WHERE id = ?", (curation_id,))
    row = cursor.fetchone()

    if not row:
        return None

    curation = dict(row)
    if curation.get("stats_json"):
        curation["stats"] = json.loads(curation["stats_json"])

    return curation


def get_active_curation(conn: sqlite3.Connection, year: int) -> Optional[dict]:
    import json

    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM curations WHERE year = ? AND is_active = 1",
        (year,)
    )
    row = cursor.fetchone()

    if not row:
        return None

    curation = dict(row)
    if curation.get("stats_json"):
        curation["stats"] = json.loads(curation["stats_json"])

    return curation


def get_all_curations(conn: sqlite3.Connection, year: int) -> List[dict]:
    import json

    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM curations WHERE year = ? ORDER BY created_at DESC",
        (year,)
    )

    curations = []
    for row in cursor.fetchall():
        curation = dict(row)
        if curation.get("stats_json"):
            curation["stats"] = json.loads(curation["stats_json"])
        curations.append(curation)

    return curations


def add_curation_photo(
    conn: sqlite3.Connection,
    curation_id: int,
    photo_id: int,
    month_slot: int
) -> int:
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO curation_photos (curation_id, photo_id, month_slot)
        VALUES (?, ?, ?)
        """,
        (curation_id, photo_id, month_slot)
    )
    return cursor.lastrowid


def add_curation_photos_batch(
    conn: sqlite3.Connection,
    curation_id: int,
    photo_assignments: List[Tuple[int, int]]
) -> None:
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO curation_photos (curation_id, photo_id, month_slot)
        VALUES (?, ?, ?)
        """,
        [(curation_id, photo_id, month_slot) for photo_id, month_slot in photo_assignments]
    )


def get_twelve_photos(conn: sqlite3.Connection, year: int) -> List[dict]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.*, cp.month_slot
        FROM curation_photos cp
        JOIN photos p ON cp.photo_id = p.id
        JOIN curations c ON cp.curation_id = c.id
        WHERE c.year = ? AND c.is_active = 1
        ORDER BY cp.month_slot
        """,
        (year,)
    )
    return [dict(row) for row in cursor.fetchall()]


def get_curation_photos(conn: sqlite3.Connection, curation_id: int) -> List[dict]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.*, cp.month_slot
        FROM curation_photos cp
        JOIN photos p ON cp.photo_id = p.id
        WHERE cp.curation_id = ?
        ORDER BY cp.month_slot
        """,
        (curation_id,)
    )
    return [dict(row) for row in cursor.fetchall()]


def save_swap(
    conn: sqlite3.Connection,
    curation_id: int,
    month_slot: int,
    old_photo_id: int,
    new_photo_id: int,
    reason: Optional[str] = None
) -> int:
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO swaps (curation_id, month_slot, old_photo_id, new_photo_id, reason)
        VALUES (?, ?, ?, ?, ?)
        """,
        (curation_id, month_slot, old_photo_id, new_photo_id, reason)
    )
    return cursor.lastrowid


def apply_swap(
    conn: sqlite3.Connection,
    curation_id: int,
    month_slot: int,
    new_photo_id: int
) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE curation_photos
        SET photo_id = ?
        WHERE curation_id = ? AND month_slot = ?
        """,
        (new_photo_id, curation_id, month_slot)
    )


def execute_swap(
    conn: sqlite3.Connection,
    curation_id: int,
    month_slot: int,
    new_photo_id: int,
    reason: Optional[str] = None
) -> None:
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT photo_id FROM curation_photos
        WHERE curation_id = ? AND month_slot = ?
        """,
        (curation_id, month_slot)
    )
    row = cursor.fetchone()

    if not row:
        raise ValueError(f"No photo found for month slot {month_slot} in curation {curation_id}")

    old_photo_id = row["photo_id"]

    if old_photo_id == new_photo_id:
        return

    save_swap(conn, curation_id, month_slot, old_photo_id, new_photo_id, reason)
    apply_swap(conn, curation_id, month_slot, new_photo_id)


def get_swap_history(conn: sqlite3.Connection, curation_id: int) -> List[dict]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT s.*,
               old.filename as old_photo_filename,
               new.filename as new_photo_filename
        FROM swaps s
        JOIN photos old ON s.old_photo_id = old.id
        JOIN photos new ON s.new_photo_id = new.id
        WHERE s.curation_id = ?
        ORDER BY s.swapped_at DESC
        """,
        (curation_id,)
    )
    return [dict(row) for row in cursor.fetchall()]


def get_curation_stats(conn: sqlite3.Connection, curation_id: int) -> dict:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            COUNT(cp.id) as photo_count,
            AVG(p.combined_score) as avg_score,
            MIN(p.combined_score) as min_score,
            MAX(p.combined_score) as max_score
        FROM curation_photos cp
        JOIN photos p ON cp.photo_id = p.id
        WHERE cp.curation_id = ?
        """,
        (curation_id,)
    )

    stats = dict(cursor.fetchone())

    cursor.execute(
        "SELECT COUNT(*) as swap_count FROM swaps WHERE curation_id = ?",
        (curation_id,)
    )
    stats["swap_count"] = cursor.fetchone()["swap_count"]

    return stats


def deactivate_curation(conn: sqlite3.Connection, curation_id: int) -> None:
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE curations SET is_active = 0 WHERE id = ?",
        (curation_id,)
    )


def activate_curation(conn: sqlite3.Connection, curation_id: int) -> None:
    cursor = conn.cursor()

    cursor.execute("SELECT year FROM curations WHERE id = ?", (curation_id,))
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Curation {curation_id} not found")

    year = row["year"]

    cursor.execute(
        "UPDATE curations SET is_active = 0 WHERE year = ? AND is_active = 1",
        (year,)
    )

    cursor.execute(
        "UPDATE curations SET is_active = 1 WHERE id = ?",
        (curation_id,)
    )


def delete_curation(conn: sqlite3.Connection, curation_id: int) -> None:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM curations WHERE id = ?", (curation_id,))
