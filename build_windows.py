"""
PyInstaller build script for Datamarkin on Windows/Linux.

Usage:
    python build_windows.py
    python build_windows.py --cuda    # include SAM2/CUDA backend
"""

import subprocess
import sys


def main():
    cuda = "--cuda" in sys.argv

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "Datamarkin",
        "--onedir",
        "--noconsole",
        "--add-data", "templates;templates",
        "--add-data", "static;static",
        "--hidden-import", "flask",
        "--hidden-import", "flask.templating",
        "--hidden-import", "jinja2",
        "--hidden-import", "sqlite3",
        "--hidden-import", "PIL",
        "--hidden-import", "webview",
        "--hidden-import", "webview.platforms.winforms",
        "--hidden-import", "sam3_backend.base",
        "--hidden-import", "sam3_backend.status",
        "--hidden-import", "sam3_backend.downloader",
    ]

    if cuda:
        cmd += [
            "--hidden-import", "sam3_backend.torch_backend",
            "--hidden-import", "torch",
            "--hidden-import", "sam2",
        ]

    cmd.append("main.py")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
