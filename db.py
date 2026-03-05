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
    """)
    conn.close()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def seed_db() -> None:
    """Insert sample projects. Additive — each call adds new rows."""
    conn = get_db()
    ts = now()

    projects = [
        ("Object Detection - Urban Scenes", "active", "object_detection", 0, "YOLOv8",
         "Training dataset for detecting vehicles, pedestrians, and traffic signs in city environments.",
         '{"batch_size": 32, "epochs": 100, "lr": 0.001}',
         '{"horizontal_flip": true, "resize": 640, "mosaic": true}',
         '{"normalize": true, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}',
         '{"car": "#ff0000", "person": "#00ff00", "traffic sign": "#ffff00", "bike": "#0000ff"}'),
        ("Medical Imaging - X-Ray", "active", "classification", 1, "ResNet50",
         "Chest X-ray images annotated for pneumonia and fracture detection.",
         '{"batch_size": 16, "epochs": 50, "lr": 0.0001}',
         '{"random_rotation": 15, "random_crop": 0.1}',
         '{"normalize": true, "mean": [127.5], "std": [127.5]}',
         '{"pneumonia": "#ff0000", "fracture": "#ffaa00", "normal": "#00ff00"}'),
        ("Satellite Imagery Analysis", "training", "segmentation", 1, "U-Net",
         "Land use classification for agricultural monitoring and urban planning.",
         '{"batch_size": 8, "epochs": 200, "lr": 0.00001}',
         '{"color_jitter": 0.2, "random_rotation": 5}',
         '{"normalize": true, "mean": [0.5], "std": [0.5]}',
         '{"forest": "#228B22", "agriculture": "#90EE90", "urban": "#808080", "water": "#4169E1"}'),
        ("Old Facial Recognition", "archived", "classification", 0, "FaceNet",
         "Archived project for facial recognition - superseded by newer model.",
         '{"batch_size": 32, "epochs": 100}',
         '{"horizontal_flip": true}',
         '{"normalize": true}',
         '{"face": "#ffffff"}'),
    ]

    for name, status, ptype, train, arch, desc, conf, aug, pre, labels in projects:
        conn.execute(
            """INSERT INTO projects
               (id, name, status, sort_order, created_at, updated_at,
                type, train, model_architecture, description,
                configuration, augmentation, preprocessing, labels)
               VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (new_id(), name, status, ts, ts, ptype, train, arch, desc, conf, aug, pre, labels),
        )

    conn.commit()
    conn.close()
