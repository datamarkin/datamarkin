"""
Unified SAM3 backend.

Works with both mlx-sam3 (Apple Silicon) and official sam3 (CUDA).
Both packages expose the same Python interface:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

Import errors propagate to the factory in sam3_backend/__init__.py.

mlx-sam3 API:
    Sam3Processor.set_image(img) -> state dict with 'backbone_out'
    Sam3Processor.add_geometric_prompt(box, label, state)
        box: [cx, cy, w, h] normalized [0, 1]
        label: True=positive, False=negative
    Sam3Processor.set_text_prompt(prompt, state)
    Sam3Processor.reset_all_prompts(state)
"""

from pathlib import Path

import numpy as np
from PIL import Image

from sam3 import build_sam3_image_model  # noqa: F401 — ImportError → UnavailableBackend
from sam3.model.sam3_image_processor import Sam3Processor

from sam3_backend.base import SAMBackend


class MLXBackend(SAMBackend):
    def __init__(self) -> None:
        self._model = None
        self._processor: Sam3Processor | None = None
        self._embeddings: dict[str, object] = {}

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            import sam3  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def load(self, model_dir: Path) -> None:
        device = _detect_device()
        # Try calling conventions from most specific to most generic.
        # Official sam3 (CUDA/MPS) takes checkpoint_path + device;
        # mlx-sam3 takes local_weights_dir and does NOT accept device/load_from_HF.
        checkpoint = _find_checkpoint(model_dir)
        attempts = [
            dict(checkpoint_path=str(checkpoint), device=device, load_from_HF=False),
            dict(local_weights_dir=str(model_dir)),
            dict(checkpoint_path=str(checkpoint)),
        ]
        errors = []
        for kwargs in attempts:
            try:
                self._model = build_sam3_image_model(**kwargs)
                break
            except TypeError as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError("Could not load SAM3 model:\n" + "\n".join(errors))
        self._processor = Sam3Processor(self._model)
        self._embeddings.clear()

    # ------------------------------------------------------------------
    def encode_image(self, image_path: Path) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        embedding_id = image_path.stem
        if embedding_id not in self._embeddings:
            img = Image.open(image_path).convert("RGB")
            state = self._processor.set_image(img)
            self._embeddings[embedding_id] = state
        return embedding_id

    # ------------------------------------------------------------------
    def predict(
        self,
        embedding_id: str,
        prompts: list[dict],
        width: int,
        height: int,
    ) -> list[dict]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if embedding_id not in self._embeddings:
            raise KeyError(f"No cached embedding for '{embedding_id}'. Call encode_image() first.")

        state = self._embeddings[embedding_id]
        self._processor.reset_all_prompts(state)

        result = None
        for p in prompts:
            if p["type"] == "point":
                # Convert point to normalized [cx, cy, w, h] box.
                # 10% box gives one clean mask for typical object sizes.
                cx, cy = float(p["x"]), float(p["y"])
                box = [cx, cy, 0.1, 0.1]
                label = p.get("polarity", 1) == 1
            elif p["type"] == "box":
                # Convert [x1, y1, x2, y2] normalized to [cx, cy, w, h]
                x1, y1 = float(p["x1"]), float(p["y1"])
                x2, y2 = float(p["x2"]), float(p["y2"])
                box = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                label = True
            else:
                continue
            result = self._processor.add_geometric_prompt(box=box, label=label, state=state)

        if result is None:
            return []
        return _extract_masks(result, width, height)

    # ------------------------------------------------------------------
    def predict_text(
        self,
        embedding_id: str,
        text_prompt: str,
        width: int,
        height: int,
        confidence_threshold: float = 0.3,
    ) -> list[dict]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if embedding_id not in self._embeddings:
            raise KeyError(f"No cached embedding for '{embedding_id}'. Call encode_image() first.")

        state = self._embeddings[embedding_id]
        self._processor.reset_all_prompts(state)

        orig_threshold = None
        if hasattr(self._processor, "confidence_threshold"):
            orig_threshold = self._processor.confidence_threshold
            self._processor.confidence_threshold = confidence_threshold

        try:
            result = self._processor.set_text_prompt(prompt=text_prompt, state=state)
        finally:
            if orig_threshold is not None:
                self._processor.confidence_threshold = orig_threshold

        return _extract_masks(result, width, height)

    # ------------------------------------------------------------------
    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._embeddings.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_masks(result: dict, width: int, height: int) -> list[dict]:
    """Convert a Sam3Processor result dict to a list of normalized mask dicts."""
    masks_raw = result.get("masks", [])
    scores_raw = result.get("scores", [])
    if not hasattr(masks_raw, "__len__") or len(masks_raw) == 0:
        return []

    masks_np = _to_numpy(masks_raw)
    scores_flat = _to_numpy(scores_raw).flatten().tolist() if len(scores_raw) > 0 else []
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]  # (N, 1, H, W) → (N, H, W)

    results = []
    for i, mask in enumerate(masks_np):
        entry = _normalize_mask(mask, width, height)
        if entry:
            entry["score"] = float(scores_flat[i]) if i < len(scores_flat) else 0.0
            results.append(entry)
    return results


def _to_numpy(x) -> np.ndarray:
    """Convert PyTorch tensor, MLX array, or ndarray to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.array(x)


def _mask_to_polygon(mask_hw: np.ndarray) -> list[float]:
    """Convert a H×W boolean/uint8 mask to a flat polygon [x, y, ...]."""
    import cv2

    mask_uint8 = (mask_hw > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2)
    return [float(v) for pair in pts for v in pair]


def _normalize_mask(mask: np.ndarray, width: int, height: int) -> dict | None:
    """Convert H×W mask to normalized segmentation dict, or None if empty."""
    polygon = _mask_to_polygon(mask)
    if not polygon:
        return None
    norm_seg = [v / width if i % 2 == 0 else v / height for i, v in enumerate(polygon)]
    xs = norm_seg[0::2]
    ys = norm_seg[1::2]
    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    return {"segmentation": norm_seg, "bbox": bbox}


def _detect_device() -> str:
    """Detect best available device: cuda → mps → cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _find_checkpoint(model_dir: Path) -> Path:
    """Return the first weight file found in model_dir."""
    for ext in (".safetensors", ".pt", ".pth", ".bin"):
        matches = list(model_dir.glob(f"*{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")
