# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Datamarkin macOS .app bundle.

Usage:
    pyinstaller Datamarkin.spec
"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── Data files ───────────────────────────────────────────────────────────────

datas = [
    ("templates", "templates"),
    ("static", "static"),
    ("scripts", "scripts"),
]

# MLX Metal shader bytecode — required for Apple Silicon GPU kernels
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

# ── Hidden imports ───────────────────────────────────────────────────────────
# Packages PyInstaller can't auto-detect (compiled extensions, dynamic imports)

hiddenimports = [
    # Flask + web
    "flask", "flask.templating", "jinja2", "markupsafe",
    "sqlite3",
    # PyWebView macOS backend
    "webview", "webview.platforms.cocoa",
    # Core ML/CV
    "PIL", "cv2", "numpy",
    "torch", "torchvision",
    "scipy", "pandas", "pyarrow",
    # Training stack
    "pytorch_lightning",
    "rfdetr", "rfdetr.training",
    "pixelflow",
    "mozo",
    # MLX (Apple Silicon)
    "mlx", "mlx.core", "mlx.nn",
    # Geometry
    "shapely",
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

# Collect all submodules for packages with heavy dynamic imports
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("torchvision")
hiddenimports += collect_submodules("rfdetr")
hiddenimports += collect_submodules("mozo")
hiddenimports += collect_submodules("pytorch_lightning")
hiddenimports += collect_submodules("lightning_utilities")
hiddenimports += collect_submodules("mlx")

try:
    hiddenimports += collect_submodules("efficient_track_anything")
except Exception:
    pass

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
    excludes=[
        # GUI toolkits we don't use
        "tkinter", "_tkinter",
        "PyQt5", "PyQt6", "PySide2", "PySide6",
        "wx",
        # Non-macOS pywebview backends
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
        "webview.platforms.mshtml",
        "webview.platforms.cef",
        "webview.platforms.gtk",
        "webview.platforms.qt",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Datamarkin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    console=False,
    # icon="static/icon.icns",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=False,
    name="Datamarkin",
)

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
