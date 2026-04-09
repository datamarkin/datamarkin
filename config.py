import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys._MEIPASS)
else:
    APP_DIR = Path(__file__).parent

APP_NAME = "Datamarkin" #duuuh
APP_VERSION = "0.1.0"

DATA_DIR = Path.home() / "Datamarkin"
DB_PATH = DATA_DIR / "datamarkin.db"
FILES_DIR = DATA_DIR / "files"
MODELS_DIR = DATA_DIR / "models"
TRAINING_JOBS_DIR = DATA_DIR / "training_jobs"
EFFICIENTTAM_MODELS_DIR = DATA_DIR / "dtmfiles" / "EfficientTAM"

FLASK_PORT = 5001

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

GITHUB_OWNER = "datamarkin"
GITHUB_REPO = "datamarkin"


def file_path(filename: str) -> Path:
    prefix = filename[:3]
    return FILES_DIR / prefix / filename
