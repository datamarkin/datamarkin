"""
py2app setup script for Datamarkin macOS .app bundle.

Usage:
    python build_macos.py py2app
"""

import os
from setuptools import setup


def collect_files(directory):
    """Recursively collect all files from a directory for py2app data_files."""
    data_files = []
    for root, dirs, files in os.walk(directory):
        file_paths = [os.path.join(root, f) for f in files]
        if file_paths:
            data_files.append((root, file_paths))
    return data_files


APP = ["main.py"]

DATA_FILES = collect_files("templates") + collect_files("static")

try:
    import mlx
    from pathlib import Path as _Path
    _mlx_dir = str(_Path(mlx.__file__).parent)
    DATA_FILES += collect_files(_mlx_dir)
except ImportError:
    _mlx_dir = None

OPTIONS = {
    "argv_emulation": False,
    "strip": True,
    "includes": ["WebKit", "Foundation", "webview"],
    "packages": ["flask", "jinja2", "sqlite3", "PIL", "mlx", "mlx_sam"],
    # "iconfile": "static/icon.icns",
}

setup(
    app=APP,
    name="Datamarkin",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
