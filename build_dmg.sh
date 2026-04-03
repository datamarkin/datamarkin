#!/bin/bash
# Create a .dmg installer from the py2app .app bundle.
#
# Usage:
#   python build_macos.py py2app   # build .app first
#   ./build_dmg.sh                 # then wrap in .dmg
#
# Requires: brew install create-dmg

set -e

VERSION=$(python -c "from config import APP_VERSION; print(APP_VERSION)")
DMG_NAME="Datamarkin-${VERSION}.dmg"
APP_PATH="dist/Datamarkin.app"

if [ ! -d "$APP_PATH" ]; then
    echo "Error: $APP_PATH not found. Run 'python build_macos.py py2app' first."
    exit 1
fi

# Remove previous DMG if exists
rm -f "$DMG_NAME"

create-dmg \
    --volname "Datamarkin" \
    --window-size 600 400 \
    --icon-size 128 \
    --icon "Datamarkin.app" 150 200 \
    --app-drop-link 450 200 \
    "$DMG_NAME" \
    "$APP_PATH"

echo "Created $DMG_NAME"
