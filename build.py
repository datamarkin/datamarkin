"""
Cross-platform build script for Datamarkin.

Usage:
    python build.py           # build for current platform
    python build.py --clean   # clean build (remove previous artifacts)
"""

import platform
import subprocess
import sys


def main():
    clean = "--clean" in sys.argv

    print(f"Building Datamarkin for {platform.system()} ({platform.machine()})")

    # Validate critical dependencies
    missing = []
    for mod in ["flask", "webview", "torch", "PIL", "pixelflow", "mozo"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print(f"ERROR: Missing dependencies: {', '.join(missing)}")
        print("Install requirements first (see requirements/ directory).")
        sys.exit(1)

    cmd = [sys.executable, "-m", "PyInstaller"]
    if clean:
        cmd.append("--clean")
    cmd.append("Datamarkin.spec")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("\nBuild complete! Output in dist/Datamarkin/")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
