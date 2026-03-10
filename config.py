from pathlib import Path

APP_NAME = "Datamarkin" #duuuh
APP_VERSION = "0.1.0"

DATA_DIR = Path.home() / "Datamarkin"
DB_PATH = DATA_DIR / "datamarkin.db"
FILES_DIR = DATA_DIR / "files"
MODELS_DIR = DATA_DIR / "models"
SAM_MODELS_DIR = MODELS_DIR / "sam3"
FLASK_PORT = 5001

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

GITHUB_OWNER = "datamarkin"
GITHUB_REPO = "datamarkin"


def file_path(filename: str) -> Path:
    prefix = filename[:3]
    return FILES_DIR / prefix / filename
