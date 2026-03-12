"""SAM3 MLX Backend - Minimal point-based implementation."""

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    build_sam3_image_model = None  # noqa: F811
    Sam3Processor = None  # noqa: F811

from sam3_backend.base import SAMBackend


class MLXBackend(SAMBackend):
    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._cache: dict[str, tuple] = {}

    def is_available(self) -> bool:
        try:
            import sam3
            return True
        except ImportError:
            return False

    def load(self, model_dir: Path) -> None:
        if build_sam3_image_model is None:
            raise RuntimeError("sam3 package not installed")

        self._model = build_sam3_image_model()
        self._processor = Sam3Processor(self._model, confidence_threshold=0.5)
        self._cache.clear()

    def encode_image(self, image_path: Path) -> str:
        if self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        embedding_id = image_path.stem
        img = Image.open(image_path).convert("RGB")
        state = self._processor.set_image(img)
        self._cache[embedding_id] = (state, img.width, img.height)
        return embedding_id

    def predict(
        self,
        embedding_id: str,
        points: list[list[float]],
        labels: list[bool],
        width: int,
        height: int,
    ) -> list[dict]:
        if self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if embedding_id not in self._cache:
            raise KeyError(f"No cached embedding for '{embedding_id}'")

        state, _, _ = self._cache[embedding_id]
        self._processor.reset_all_prompts(state)

        state = self._processor.add_points_prompt(
            points=points, labels=labels, state=state
        )

        masks_raw = state.get("masks", [])
        boxes_raw = state.get("boxes", [])
        scores_raw = state.get("scores", [])

        return _extract_masks(masks_raw, boxes_raw, scores_raw, width, height)

    def unload(self) -> None:
        self._cache.clear()
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        import gc

        gc.collect()


def _extract_masks(
    masks_raw, boxes_raw, scores_raw, width: int, height: int
) -> list[dict]:
    """Convert raw SAM3 output to normalized mask dicts."""
    masks_np = _to_numpy(masks_raw)
    scores_flat = (
        _to_numpy(scores_raw).flatten().tolist() if len(scores_raw) > 0 else []
    )

    if not hasattr(masks_np, "__len__") or len(masks_np) == 0:
        return []

    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]

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
    """Convert HxW boolean/uint8 mask to flat polygon [x, y, ...]."""
    import cv2

    mask_uint8 = (mask_hw > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2)
    return [float(v) for pair in pts for v in pair]


def _normalize_mask(mask: np.ndarray, width: int, height: int) -> dict | None:
    """Convert HxW mask to normalized segmentation dict, or None if empty."""
    polygon = _mask_to_polygon(mask)
    if not polygon:
        return None

    norm_seg = [v / width if i % 2 == 0 else v / height for i, v in enumerate(polygon)]
    xs = norm_seg[0::2]
    ys = norm_seg[1::2]
    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

    return {"segmentation": norm_seg, "bbox": bbox}
