from pathlib import Path
from PIL import Image, ImageOps
from config import DATA_DIR

THUMBS_DIR = DATA_DIR / "thumbs"

PRESETS = {
    "small":       {"op": "contain", "size": (480, 480), "save": True},
    "medium":      {"op": "contain", "size": (300, 300), "save": True},
    "cover_small": {"op": "cover",   "size": (100, 100), "save": True},
    "square":      {"op": "fit",     "size": (200, 200), "save": True},
    "original":    {"op": None,      "size": None,   "save": False},  # On-demand as JPG
}


def thumb_path(file_id: str, preset: str) -> Path:
    return THUMBS_DIR / preset / f"{file_id}.jpg"


def get_or_create_thumb(source_path: Path, file_id: str, preset: str):
    cfg = PRESETS[preset]

    if not cfg["save"]:
        # Process in memory and return PIL Image for on-demand serving
        img = Image.open(source_path)
        img = img.convert("RGB")
        return img

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
