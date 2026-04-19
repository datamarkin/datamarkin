# Building Datamarkin

Datamarkin uses PyInstaller to produce standalone desktop apps. A single unified `Datamarkin.spec` auto-detects the platform at build time.

## Quick Reference

| Platform | Build | Package | Output |
|----------|-------|---------|--------|
| macOS | `python build.py` | `./build_dmg.sh` | `Datamarkin-{version}.dmg` |
| Windows | `python build.py` | `iscc installer.iss` | `Datamarkin-{version}-Setup.exe` |
| Linux | `python build.py` | `./build_tar.sh` | `Datamarkin-{version}-linux-x86_64.tar.gz` |

## Prerequisites

### All Platforms

- Python 3.12
- A **pure pip venv** (not conda — see note below)

### macOS (Apple Silicon)

```bash
python3.12 -m venv ~/datamarkin-venv
source ~/datamarkin-venv/bin/activate

pip install -r requirements/macos.txt
pip install torch torchvision
pip install rfdetr detectron2 pytorch_lightning albumentations
pip install efficient_track_anything falcon-perception
pip install pixelflow mozo agentui
pip install pyinstaller
```

EfficientTAM config fix (required — see CLAUDE.md for details):

```bash
cp -r /path/to/EfficientTAM/efficient_track_anything/configs \
      ~/datamarkin-venv/lib/python3.12/site-packages/efficient_track_anything/configs
```

### Windows

```powershell
python -m venv C:\Users\<user>\datamarkin-venv
C:\Users\<user>\datamarkin-venv\Scripts\Activate.ps1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements\base.txt
pip install sam3 rfdetr pytorch_lightning albumentations
pip install efficient_track_anything falcon-perception
pip install pixelflow mozo agentui
pip install pyinstaller
```

Inno Setup is required for the installer: https://jrsoftware.org/isinfo.php

### Ubuntu / Linux

System packages for PyWebView GTK backend:

```bash
sudo apt install python3-dev python3-gi python3-gi-cairo \
    gir1.2-webkit2-4.1 libgirepository1.0-dev
```

Python environment:

```bash
python3.12 -m venv ~/datamarkin-venv
source ~/datamarkin-venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/base.txt
pip install sam3 rfdetr pytorch_lightning albumentations
pip install efficient_track_anything falcon-perception
pip install pixelflow mozo agentui
pip install pyinstaller
```

## Building

From the project root, with your venv activated:

```bash
python build.py
```

This validates dependencies and runs `pyinstaller Datamarkin.spec`. Add `--clean` for a fresh build.

Output lands in `dist/Datamarkin/`.

## Packaging

### macOS — DMG

Requires `create-dmg` (`brew install create-dmg`):

```bash
./build_dmg.sh
```

### Windows — Installer

Requires Inno Setup installed and `iscc` on PATH:

```powershell
iscc installer.iss
```

Output: `Output/Datamarkin-{version}-Setup.exe`

### Linux — tar.gz

```bash
./build_tar.sh
```

## Platform Differences

| Feature | macOS | Windows | Linux |
|---------|-------|---------|-------|
| WebView engine | WebKit (Cocoa) | Edge WebView2 | WebKitGTK |
| GPU backend | MPS (Metal) / MLX | CUDA | CUDA |
| Falcon Perception | MLX backend | PyTorch backend | PyTorch backend |
| SAM3 | mlx_sam3 | sam3 (torch) | sam3 (torch) |
| Process signals | SIGTERM (graceful) | TerminateProcess (hard kill) | SIGTERM (graceful) |
| Training dataset prep | Symlinks | File copy | Symlinks |

## Why Not Conda?

PyInstaller needs to trace imports and copy shared libraries from `site-packages`. Conda installs shared libs into `$CONDA_PREFIX/lib/` with `@rpath` references that PyInstaller cannot resolve. The bundled app will crash with missing dylib errors. Always use a `python -m venv` environment.

## Troubleshooting

**Missing DLLs / dylibs at runtime**: A dependency's shared library wasn't collected. Add it to the `binaries` list in `Datamarkin.spec` or add the package to `hiddenimports`.

**Import errors in frozen app**: The module uses dynamic imports that PyInstaller can't trace. Add it to `hiddenimports` in the spec file, or use `collect_submodules("package_name")`.

**WebView doesn't launch (Linux)**: Ensure GTK system packages are installed (`python3-gi`, `gir1.2-webkit2-4.1`).

**WebView doesn't launch (Windows)**: Edge WebView2 Runtime is required. It's pre-installed on Windows 10/11 but may be missing on older systems. Download from https://developer.microsoft.com/en-us/microsoft-edge/webview2/
