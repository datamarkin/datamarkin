"""
SAM3 backend factory.

Selects the appropriate backend based on the current platform:
- darwin  → MLX backend (Apple Silicon, uses mlx-sam3 library)
- other   → Torch/CUDA backend

Backends are only instantiated inside try/except ImportError so missing
frameworks fail silently. The app never crashes on startup.
"""

import sys

from sam3_backend.base import SAMBackend, UnavailableBackend
from sam3_backend.status import get_sam_status

_backend: SAMBackend | None = None


def get_sam_backend() -> SAMBackend:
    """Return the singleton backend instance (lazy-initialised)."""
    global _backend
    if _backend is not None:
        return _backend

    if sys.platform == "darwin":
        try:
            from sam3_backend.mlx_backend import MLXBackend
            _backend = MLXBackend()
        except ImportError:
            _backend = UnavailableBackend("mlx-sam3 not installed")
    else:
        try:
            from sam3_backend.torch_backend import TorchBackend
            _backend = TorchBackend()
        except ImportError:
            _backend = UnavailableBackend("torch / sam-2 not installed")

    return _backend


__all__ = ["get_sam_backend", "get_sam_status"]
