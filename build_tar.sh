#!/bin/bash
# Create a .tar.gz archive from the PyInstaller build for Linux.
#
# Usage:
#   python build.py        # build first
#   ./build_tar.sh         # then package
#

set -e

VERSION=$(python3 -c "from config import APP_VERSION; print(APP_VERSION)")
TAR_NAME="Datamarkin-${VERSION}-linux-x86_64.tar.gz"
DIST_DIR="dist/Datamarkin"

if [ ! -d "$DIST_DIR" ]; then
    echo "Error: $DIST_DIR not found. Run 'python build.py' first."
    exit 1
fi

rm -f "$TAR_NAME"

tar -czf "$TAR_NAME" -C dist Datamarkin

echo "Created $TAR_NAME"
