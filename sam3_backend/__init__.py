"""
SAM3 backend factory.

Single unified backend for all platforms (macOS/CUDA/CPU).
Install mlx-sam3 on Apple Silicon or sam3 on CUDA — both expose the same API.
"""

from sam3_backend.base import SAMBackend, UnavailableBackend
from sam3_backend.status import get_sam_status

_backend: SAMBackend | None = None


def get_sam_backend() -> SAMBackend:
    """Return the singleton backend instance (lazy-initialised)."""
    global _backend
    if _backend is not None:
        return _backend

    try:
        from sam3_backend.mlx_backend import MLXBackend
        _backend = MLXBackend()
    except ImportError:
        _backend = UnavailableBackend("sam3 not installed (pip install mlx-sam3 or sam3)")

    return _backend


__all__ = ["get_sam_backend", "get_sam_status"]
