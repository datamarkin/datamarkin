from pathlib import Path
from PIL import Image, ImageOps
from config import DATA_DIR

THUMBS_DIR = DATA_DIR / "thumbs"

PRESETS = {
    "small":       {"op": "contain", "size": (100, 100)},
    "medium":      {"op": "contain", "size": (300, 300)},
    "cover_small": {"op": "cover",   "size": (100, 100)},
    "square":      {"op": "fit",     "size": (200, 200)},
}


def thumb_path(file_id: str, preset: str) -> Path:
    return THUMBS_DIR / preset / f"{file_id}.jpg"


def get_or_create_thumb(source_path: Path, file_id: str, preset: str) -> Path:
    cfg = PRESETS[preset]
    dest = thumb_path(file_id, preset)

    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(source_path)
    img = img.convert("RGB")

    op = cfg["op"]
    size = cfg["size"]

    if op == "contain":
        img = ImageOps.contain(img, size)
    elif op == "cover":
        img = ImageOps.cover(img, size)
    elif op == "fit":
        img = ImageOps.fit(img, size)

    img.save(dest, "JPEG", quality=85)
    return dest
