import sqlite3
import uuid
from datetime import datetime, timezone

from config import DATA_DIR, DB_PATH, FILES_DIR


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "projects").mkdir(exist_ok=True)
    FILES_DIR.mkdir(exist_ok=True)

    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            status        TEXT DEFAULT 'active',
            sort_order    INTEGER DEFAULT 0,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            type          TEXT NOT NULL,
            train         INTEGER DEFAULT 0,
            model_architecture TEXT,
            description   TEXT,
            configuration TEXT,
            augmentation  TEXT,
            preprocessing TEXT,
            labels        TEXT
        );

        CREATE TABLE IF NOT EXISTS files (
            id              TEXT PRIMARY KEY,
            project_id      TEXT REFERENCES projects(id) ON DELETE CASCADE,
            filename        TEXT NOT NULL,
            extension       TEXT,
            width           INTEGER,
            height          INTEGER,
            filesize        INTEGER,
            checksum        TEXT,
            is_annotated    INTEGER DEFAULT 0,
            split           TEXT,
            sort_order      INTEGER DEFAULT 0,
            annotations     TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );
    """)
    conn.close()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())
