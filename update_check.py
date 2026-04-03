"""
Background version check against GitHub Releases.

Usage:
    # Start the check (call once from a daemon thread):
    threading.Thread(target=check_for_update, daemon=True).start()

    # Query the cached result (from any thread):
    info = get_update_info()  # None or {"version", "url", "download_url"}
"""

import json
import platform
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from packaging.version import Version, InvalidVersion

from config import APP_VERSION, GITHUB_OWNER, GITHUB_REPO


def _asset_suffix():
    """Return the expected file extension for this platform's release asset."""
    if sys.platform == "darwin":
        return ".dmg"
    elif sys.platform == "win32":
        return ".exe"
    else:
        return ".tar.gz"

_latest = None
_lock = threading.Lock()

_API_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"


def check_for_update():
    """Check GitHub for a newer release. Runs in a background thread."""
    global _latest
    time.sleep(5)
    try:
        req = urllib.request.Request(_API_URL, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        tag = data.get("tag_name", "")
        remote_version = tag.lstrip("v")
        if Version(remote_version) <= Version(APP_VERSION):
            return

        # Find the asset matching this platform (.dmg / .exe / .tar.gz)
        suffix = _asset_suffix()
        download_url = None
        for asset in data.get("assets", []):
            if asset.get("name", "").endswith(suffix):
                download_url = asset["browser_download_url"]
                break

        with _lock:
            _latest = {
                "version": remote_version,
                "url": data.get("html_url", ""),
                "download_url": download_url,
            }
    except Exception:
        pass


def get_update_info():
    """Return cached update info, or None if up to date / not checked yet."""
    with _lock:
        return _latest


def download_update():
    """Download the .dmg to ~/Downloads and reveal in Finder. Returns the path."""
    info = get_update_info()
    if not info or not info.get("download_url"):
        return None

    suffix = _asset_suffix()
    dest = Path.home() / "Downloads" / f"Datamarkin-{info['version']}{suffix}"
    req = urllib.request.Request(info["download_url"])
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)

    # Reveal in file manager
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(dest)])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(dest)])
    else:
        subprocess.Popen(["xdg-open", str(dest.parent)])

    return str(dest)
