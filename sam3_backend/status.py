"""
Cheap probe: are the framework and weights available?

This module must NOT import any ML framework at module level so that
`get_sam_status()` is always fast and never crashes.
"""

import sys
from pathlib import Path


def _framework_importable() -> bool:
    """Check whether the expected ML framework package is installed."""
    from importlib.metadata import distribution, PackageNotFoundError
    if sys.platform == "darwin":
        try:
            distribution("mlx-sam3")
            return True
        except PackageNotFoundError:
            return False
    else:
        try:
            import importlib.util
            return importlib.util.find_spec("sam2") is not None
        except Exception:
            return False


def _weights_present(sam_models_dir: Path) -> dict[str, bool]:
    """Return a dict mapping variant names to whether their weights exist."""
    result: dict[str, bool] = {}
    if not sam_models_dir.exists():
        return result
    for variant_dir in sam_models_dir.iterdir():
        if variant_dir.is_dir():
            has_weights = any(
                f.suffix in (".safetensors", ".pt", ".pth", ".bin")
                for f in variant_dir.iterdir()
            )
            result[variant_dir.name] = has_weights
    return result


def get_sam_status(sam_models_dir: Path) -> dict:
    """
    Return a status dict with no ML imports.

    {
        "framework_available": bool,
        "platform": "darwin" | "other",
        "backend": "mlx" | "torch",
        "variants": {"tiny": true, "small": false, ...},
        "ready": bool   # True iff framework importable AND at least one variant ready
    }
    """
    framework_ok = _framework_importable()
    variants = _weights_present(sam_models_dir)
    any_weights = any(variants.values()) if variants else False

    return {
        "framework_available": framework_ok,
        "platform": sys.platform,
        "backend": "mlx" if sys.platform == "darwin" else "torch",
        "variants": variants,
        "ready": framework_ok and any_weights,
    }
