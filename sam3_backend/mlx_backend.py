"""
Unified SAM3 backend.

Works with both mlx-sam3 (Apple Silicon) and official sam3 (CUDA).
Both packages expose the same Python interface:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

Import errors propagate to the factory in sam3_backend/__init__.py.
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
        # Official sam3 uses checkpoint_path (file); mlx-sam3 uses local_weights_dir (dir).
        # Try checkpoint_path first; fall back on TypeError.
        try:
            checkpoint = _find_checkpoint(model_dir)
            self._model = build_sam3_image_model(
                checkpoint_path=str(checkpoint),
                device=device,
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
        except TypeError:
            # mlx-sam3 does not accept checkpoint_path
            self._model = build_sam3_image_model(
                local_weights_dir=str(model_dir),
                device=device,
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
        self._processor = Sam3Processor(self._model)
        self._embeddings.clear()

    # ------------------------------------------------------------------
    def encode_image(self, image_path: Path) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        embedding_id = image_path.stem
        if embedding_id not in self._embeddings:
            img = Image.open(image_path).convert("RGB")
            inference_state = self._processor.set_image(img)
            self._embeddings[embedding_id] = inference_state
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

        inference_state = self._embeddings[embedding_id]

        point_coords: list[list[float]] = []
        point_labels: list[int] = []
        box = None

        for p in prompts:
            if p["type"] == "point":
                point_coords.append([p["x"] * width, p["y"] * height])
                point_labels.append(p.get("polarity", 1))
            elif p["type"] == "box":
                box = np.array([
                    p["x1"] * width, p["y1"] * height,
                    p["x2"] * width, p["y2"] * height,
                ])

        point_coords_arr = np.array(point_coords) if point_coords else None
        point_labels_arr = np.array(point_labels) if point_labels else None
        box_arr = box[None, :] if box is not None else None

        masks, scores, _ = self._model.predict_inst(
            inference_state,
            point_coords=point_coords_arr,
            point_labels=point_labels_arr,
            box=box_arr,
            mask_input=None,
            multimask_output=False,
        )

        masks_np = _to_numpy(masks)
        scores_np = _to_numpy(scores)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]  # (N, 1, H, W) → (N, H, W)

        results = []
        for mask, score in zip(masks_np, scores_np):
            entry = _normalize_mask(mask, width, height)
            if entry:
                entry["score"] = float(score)
                results.append(entry)
        return results

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

        inference_state = self._embeddings[embedding_id]

        # Temporarily override confidence threshold if the processor exposes it
        orig_threshold = None
        if hasattr(self._processor, "confidence_threshold"):
            orig_threshold = self._processor.confidence_threshold
            self._processor.confidence_threshold = confidence_threshold

        try:
            self._processor.reset_all_prompts(inference_state)
            result = self._processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        finally:
            if orig_threshold is not None:
                self._processor.confidence_threshold = orig_threshold

        masks = result.get("masks", [])
        scores = result.get("scores", [])

        if len(masks) == 0:
            return []

        masks_np = _to_numpy(masks)
        scores_flat = _to_numpy(scores).flatten().tolist() if len(scores) > 0 else []
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]  # (N, 1, H, W) → (N, H, W)

        results = []
        for i, mask in enumerate(masks_np):
            entry = _normalize_mask(mask, width, height)
            if entry:
                entry["score"] = float(scores_flat[i]) if i < len(scores_flat) else 0.0
                entry["label"] = text_prompt
                results.append(entry)
        return results

    # ------------------------------------------------------------------
    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._embeddings.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
