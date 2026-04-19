# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Datamarkin — cross-platform.

Builds on macOS (.app), Windows (.exe), and Linux automatically
by detecting sys.platform at build time.

Usage:
    pyinstaller Datamarkin.spec
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

# ── Data files ───────────────────────────────────────────────────────────────

datas = [
    ("templates", "templates"),
    ("static", "static"),
    ("scripts", "scripts"),
]

# MLX Metal shader bytecode — required for Apple Silicon GPU kernels (macOS only)
if IS_MAC:
    try:
        import mlx
        mlx_dir = Path(mlx.__path__[0])
        metallib_dir = mlx_dir / "lib"
        if metallib_dir.is_dir():
            datas.append((str(metallib_dir), "mlx/lib"))
    except (ImportError, IndexError):
        pass

# EfficientTAM Hydra YAML configs (not included by pip install due to packaging bug)
try:
    import efficient_track_anything
    eta_dir = Path(efficient_track_anything.__file__).parent
    configs_dir = eta_dir / "configs"
    if configs_dir.is_dir():
        datas.append((str(configs_dir), "efficient_track_anything/configs"))
except ImportError:
    pass

# PyTorch Lightning data files (version.info, etc.)
try:
    datas += collect_data_files("pytorch_lightning")
except Exception:
    pass

# AgentUI static assets
try:
    datas += collect_data_files("agentui")
except Exception:
    pass

# RF-DETR .py source files — TorchScript JIT needs source access to compile
# functions like batch_dice_loss; bytecode-only (.pyc) bundles fail at runtime.
try:
    import rfdetr as _rfdetr
    _rfdetr_dir = Path(_rfdetr.__path__[0])
    datas.append((str(_rfdetr_dir), "rfdetr"))
except ImportError:
    pass

# Detectron2 model zoo YAML configs + base configs (needed by model_zoo.get_config_file())
try:
    import detectron2
    d2_dir = Path(detectron2.__path__[0])
    datas.append((str(d2_dir / "model_zoo"), "detectron2/model_zoo"))
    datas.append((str(d2_dir / "config"), "detectron2/config"))
except ImportError:
    pass

# ── Hidden imports ───────────────────────────────────────────────────────────
# Packages PyInstaller can't auto-detect (compiled extensions, dynamic imports)

hiddenimports = [
    # Flask + web
    "flask", "flask.templating", "jinja2", "markupsafe",
    "sqlite3",
    # PyWebView (platform backend added below)
    "webview",
    # Core ML/CV
    "PIL", "cv2", "numpy",
    "torch", "torchvision",
    "scipy", "pandas", "pyarrow",
    # Training stack
    "pytorch_lightning",
    "rfdetr", "rfdetr.training",
    "pixelflow",
    "mozo",
    # Geometry
    "shapely",
    # Detectron2
    "detectron2", "fvcore", "iopath",
    # Data processing
    "pycocotools", "matplotlib", "contourpy", "kiwisolver", "fontTools",
    # Serialization
    "safetensors", "tokenizers", "orjson", "yaml", "regex",
    # Networking
    "aiohttp", "multidict", "frozenlist", "yarl", "propcache",
    "charset_normalizer",
    # Pydantic
    "pydantic_core",
    # HuggingFace
    "xxhash", "hf_transfer", "hf_xet",
    # WebSockets
    "websockets",
    # EfficientTAM / Hydra
    "efficient_track_anything",
    "hydra", "omegaconf",
    # Augmentation pipeline
    "albumentations",
    # Update checker
    "packaging", "packaging.version",
    # App modules (imported dynamically or as subprocess)
    "scripts.training_worker",
    "pipeline", "config", "db", "db_models", "queries", "messenger",
    "thumbnails", "update_check",
]

# Platform-specific PyWebView backend
if IS_MAC:
    hiddenimports.append("webview.platforms.cocoa")
elif IS_WIN:
    hiddenimports.append("webview.platforms.edgechromium")
else:
    hiddenimports.append("webview.platforms.gtk")

# MLX (Apple Silicon — macOS only)
if IS_MAC:
    hiddenimports += ["mlx", "mlx.core", "mlx.nn"]

# Collect all submodules for packages with heavy dynamic imports
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("torchvision")
hiddenimports += collect_submodules("rfdetr")
hiddenimports += collect_submodules("mozo")
hiddenimports += collect_submodules("pytorch_lightning")
hiddenimports += collect_submodules("lightning_utilities")

if IS_MAC:
    try:
        hiddenimports += collect_submodules("mlx")
    except Exception:
        pass

hiddenimports += collect_submodules("detectron2")
hiddenimports += collect_submodules("fvcore")
hiddenimports += collect_submodules("iopath")

try:
    hiddenimports += collect_submodules("efficient_track_anything")
except Exception:
    pass

# ── Excludes ─────────────────────────────────────────────────────────────────

excludes = [
    # GUI toolkits we don't use
    "tkinter", "_tkinter",
    "PyQt5", "PyQt6", "PySide2", "PySide6",
    "wx",
]

# Exclude non-active pywebview backends
_all_webview_backends = [
    "webview.platforms.cocoa",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "webview.platforms.mshtml",
    "webview.platforms.cef",
    "webview.platforms.gtk",
    "webview.platforms.qt",
]

if IS_MAC:
    _active_backend = "webview.platforms.cocoa"
elif IS_WIN:
    _active_backend = "webview.platforms.edgechromium"
else:
    _active_backend = "webview.platforms.gtk"

excludes += [b for b in _all_webview_backends if b != _active_backend]

# Exclude MLX on non-macOS (not available)
if not IS_MAC:
    excludes += ["mlx", "mlx.core", "mlx.nn"]

# ── Analysis ─────────────────────────────────────────────────────────────────

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── Icon ─────────────────────────────────────────────────────────────────────

_icon = None
if IS_WIN and Path("static/favicon/favicon.ico").exists():
    _icon = "static/favicon/favicon.ico"
# macOS: uncomment when .icns is available
# if IS_MAC:
#     _icon = "static/icon.icns"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Datamarkin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=not IS_WIN,
    upx=False,
    console=False,
    icon=_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=not IS_WIN,
    upx=False,
    name="Datamarkin",
)

# macOS .app bundle (not used on Windows/Linux)
if IS_MAC:
    app = BUNDLE(
        coll,
        name="Datamarkin.app",
        # icon="static/icon.icns",
        bundle_identifier="com.datamarkin.app",
        info_plist={
            "CFBundleName": "Datamarkin",
            "CFBundleDisplayName": "Datamarkin",
            "CFBundleVersion": "0.1.0",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
