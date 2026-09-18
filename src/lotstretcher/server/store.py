"""
SQLite-backed vehicle registry + image-submission job tracking for server
mode. Same one-connection-per-call pattern as imaging/dedupe.py's photo
cache -- SQLite handles concurrent readers fine, and a short-lived
connection per call sidesteps cross-thread sharing issues without a
connection pool this scale doesn't need.

Two independent things tracked here, matching CarCutter's own API shape:
  vehicles          -- POST/GET/DELETE /vehicle/* register/query a vehicle
                        record; bookkeeping only, no image processing.
  image submissions -- POST /vehicle/image/submission kicks off async
                        processing of up to 60 images; status/result are
                        polled per submission_id, or delivered via
                        webhook_url when the whole batch finishes.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

DB_FILENAME = "lotstretcher-server.db"

STATUS_NEW = "new"
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _connect(data_dir: Path) -> sqlite3.Connection:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(data_dir / DB_FILENAME)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id TEXT PRIMARY KEY,
            vehicle_id TEXT,
            webhook_url TEXT,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS submission_images (
            submission_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            url TEXT NOT NULL,
            status TEXT NOT NULL,
            output_path TEXT,
            category TEXT,
            warning TEXT,
            PRIMARY KEY (submission_id, image_id)
        )
    """)
    conn.row_factory = sqlite3.Row
    return conn


# -- Vehicles ---------------------------------------------------------------

def upsert_vehicle(data_dir: Path, vehicle_id: str, metadata: dict) -> None:
    now = time.time()
    with _connect(data_dir) as conn:
        conn.execute(
            """INSERT INTO vehicles (vehicle_id, status, metadata_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(vehicle_id) DO UPDATE SET
                   metadata_json = excluded.metadata_json, updated_at = excluded.updated_at""",
            (vehicle_id, STATUS_NEW, json.dumps(metadata), now, now),
        )


def get_vehicle(data_dir: Path, vehicle_id: str) -> dict | None:
    with _connect(data_dir) as conn:
        row = conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,)).fetchone()
    return _vehicle_row_to_dict(row) if row else None


def list_vehicles(data_dir: Path, status: str | None = None) -> list[dict]:
    with _connect(data_dir) as conn:
        if status:
            rows = conn.execute("SELECT * FROM vehicles WHERE status = ? ORDER BY created_at",
                                 (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM vehicles ORDER BY created_at").fetchall()
    return [_vehicle_row_to_dict(r) for r in rows]


def set_vehicle_status(data_dir: Path, vehicle_id: str, status: str) -> bool:
    with _connect(data_dir) as conn:
        cur = conn.execute("UPDATE vehicles SET status = ?, updated_at = ? WHERE vehicle_id = ?",
                            (status, time.time(), vehicle_id))
        return cur.rowcount > 0


def delete_vehicle(data_dir: Path, vehicle_id: str) -> bool:
    with _connect(data_dir) as conn:
        cur = conn.execute("DELETE FROM vehicles WHERE vehicle_id = ?", (vehicle_id,))
        return cur.rowcount > 0


def _vehicle_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["metadata"] = json.loads(d.pop("metadata_json"))
    return d


# -- Image submissions --------------------------------------------------------

def create_submission(data_dir: Path, vehicle_id: str | None, image_urls: dict[str, str],
                       webhook_url: str | None) -> str:
    """image_urls: {image_id: url}. Returns the new submission_id."""
    submission_id = str(uuid.uuid4())
    now = time.time()
    with _connect(data_dir) as conn:
        conn.execute("INSERT INTO submissions (submission_id, vehicle_id, webhook_url, created_at) "
                     "VALUES (?, ?, ?, ?)", (submission_id, vehicle_id, webhook_url, now))
        conn.executemany(
            "INSERT INTO submission_images (submission_id, image_id, url, status) VALUES (?, ?, ?, ?)",
            [(submission_id, image_id, url, STATUS_PENDING) for image_id, url in image_urls.items()],
        )
    return submission_id


def update_image_result(data_dir: Path, submission_id: str, image_id: str, status: str,
                         output_path: str | None = None, category: str | None = None,
                         warning: str | None = None) -> None:
    with _connect(data_dir) as conn:
        conn.execute(
            "UPDATE submission_images SET status = ?, output_path = ?, category = ?, warning = ? "
            "WHERE submission_id = ? AND image_id = ?",
            (status, output_path, category, warning, submission_id, image_id),
        )


def get_submission(data_dir: Path, submission_id: str) -> dict | None:
    with _connect(data_dir) as conn:
        sub = conn.execute("SELECT * FROM submissions WHERE submission_id = ?",
                            (submission_id,)).fetchone()
        if sub is None:
            return None
        images = conn.execute("SELECT * FROM submission_images WHERE submission_id = ?",
                               (submission_id,)).fetchall()
    return {**dict(sub), "images": [dict(r) for r in images]}
