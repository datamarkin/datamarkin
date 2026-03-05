import sqlite3
import uuid
from datetime import datetime, timezone

from config import DATA_DIR, DB_PATH


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "projects").mkdir(exist_ok=True)

    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT,
            task_type   TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS images (
            id           TEXT PRIMARY KEY,
            project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            filename     TEXT NOT NULL,
            width        INTEGER,
            height       INTEGER,
            is_annotated INTEGER DEFAULT 0,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS labels (
            id         TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            name       TEXT NOT NULL,
            color      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS annotations (
            id              TEXT PRIMARY KEY,
            image_id        TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            label_id        TEXT REFERENCES labels(id) ON DELETE SET NULL,
            type            TEXT NOT NULL,
            data_json       TEXT NOT NULL,
            created_at      TEXT NOT NULL
        );
    """)
    conn.close()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())
