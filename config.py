from pathlib import Path

APP_VERSION = "0.1.0"

DATA_DIR = Path.home() / "Datamarkin"
DB_PATH = DATA_DIR / "datamarkin.db"
FILES_DIR = DATA_DIR / "files"
FLASK_PORT = 5001


def file_path(filename: str) -> Path:
    prefix = filename[:3]
    return FILES_DIR / prefix / filename
