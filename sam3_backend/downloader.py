"""
Model weight downloader — stdlib only (urllib.request + threading).

Downloads to a .tmp file then atomically renames on completion so that
partial downloads are never mistaken for complete weights.

Usage:
    dl = Downloader()
    dl.start("tiny", url, dest_dir)
    status = dl.progress("tiny")   # {"state": "downloading", "pct": 42.1}
"""

import os
import threading
import urllib.request
from pathlib import Path


# Per-variant download state: {"state": str, "pct": float, "error": str|None}
_state: dict[str, dict] = {}
_lock = threading.Lock()


# Known SAM variant download URLs (update as releases change)
VARIANT_URLS: dict[str, str] = {
    # SAM3 / Apple Silicon (mlx-sam3, single model ~3.4 GB)
    "sam3": (
        "https://huggingface.co/mlx-community/sam3-image/resolve/main/model.safetensors"
    ),
    # SAM2 / CUDA weights (official Facebook)
    "sam2-tiny": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    ),
    "sam2-small": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    ),
    "sam2-base": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
    ),
    "sam2-large": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    ),
}


def start(variant: str, dest_dir: Path, url: str | None = None) -> None:
    """
    Start a background download for the given variant.

    If url is None, it is looked up in VARIANT_URLS.
    Raises ValueError for unknown variants when url is not provided.
    """
    if url is None:
        if variant not in VARIANT_URLS:
            raise ValueError(f"Unknown SAM variant '{variant}'. Provide a URL explicitly.")
        url = VARIANT_URLS[variant]

    with _lock:
        existing = _state.get(variant, {})
        if existing.get("state") == "downloading":
            return  # already in progress

        _state[variant] = {"state": "downloading", "pct": 0.0, "error": None}

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0] or "model.safetensors"
    dest_file = dest_dir / filename
    tmp_file = dest_dir / (filename + ".tmp")

    thread = threading.Thread(
        target=_download_worker,
        args=(variant, url, tmp_file, dest_file),
        daemon=True,
        name=f"sam-download-{variant}",
    )
    thread.start()


def progress(variant: str) -> dict:
    """Return current download state for a variant."""
    with _lock:
        return dict(_state.get(variant, {"state": "idle", "pct": 0.0, "error": None}))


def cancel(variant: str) -> None:
    """Mark a download as cancelled (the thread will notice on next chunk)."""
    with _lock:
        if _state.get(variant, {}).get("state") == "downloading":
            _state[variant]["state"] = "cancelled"


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _download_worker(variant: str, url: str, tmp_path: Path, dest_path: Path) -> None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Datamarkin/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0

            with open(tmp_path, "wb") as f:
                while True:
                    with _lock:
                        state = _state.get(variant, {}).get("state")
                    if state == "cancelled":
                        return

                    chunk = resp.read(1024 * 256)  # 256 KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    pct = (downloaded / total * 100) if total else 0.0
                    with _lock:
                        if _state.get(variant, {}).get("state") == "downloading":
                            _state[variant]["pct"] = round(pct, 1)

        # Atomic rename
        os.replace(tmp_path, dest_path)

        with _lock:
            _state[variant] = {"state": "complete", "pct": 100.0, "error": None}

    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        with _lock:
            _state[variant] = {"state": "error", "pct": 0.0, "error": str(exc)}
