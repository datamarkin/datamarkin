"""
Apple Silicon / MLX SAM3 backend.

Wraps the mlx-sam3 library (importable as `sam3`). This file is only
imported on darwin; import errors are caught by the factory in
sam3_backend/__init__.py.
"""

from pathlib import Path

# Existence check — ImportError propagates to factory → UnavailableBackend.
from sam3.model_builder import build_sam3_image_model  # noqa: F401

from sam3_backend.base import SAMBackend


class MLXBackend(SAMBackend):
    def __init__(self) -> None:
        self._model = None
        self._embeddings: dict[str, object] = {}
        self._model_dir: Path | None = None

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            from importlib.metadata import distribution
            distribution("mlx-sam3")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    def load(self, model_dir: Path) -> None:
        """Load SAM3 weights from model_dir."""
        from sam3.model_builder import build_sam3_image_model
        self._model = build_sam3_image_model(local_weights_dir=str(model_dir))
        self._model_dir = model_dir
        self._embeddings.clear()

    # ------------------------------------------------------------------
    def encode_image(self, image_path: Path) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        embedding_id = image_path.stem

        if embedding_id not in self._embeddings:
            import mlx.core as mx
            from PIL import Image
            import numpy as np

            img = Image.open(image_path).convert("RGB")
            img_array = mx.array(np.array(img))

            # mlx_sam3 API: model.encode_image(image_array)
            embedding = self._model.encode_image(img_array)
            self._embeddings[embedding_id] = embedding

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

        embedding = self._embeddings[embedding_id]

        # Convert normalized prompts → pixel coords for mlx_sam
        pixel_prompts = _denormalize_prompts(prompts, width, height)

        # mlx_sam3 API: model.predict(embedding, prompts) → list of mask results
        results = self._model.predict(embedding, pixel_prompts)

        return _normalize_results(results, width, height)

    # ------------------------------------------------------------------
    def unload(self) -> None:
        self._model = None
        self._embeddings.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _denormalize_prompts(prompts: list[dict], w: int, h: int) -> list[dict]:
    out = []
    for p in prompts:
        if p["type"] == "point":
            out.append({
                "type": "point",
                "x": p["x"] * w,
                "y": p["y"] * h,
                "polarity": p.get("polarity", 1),
            })
        elif p["type"] == "box":
            out.append({
                "type": "box",
                "x1": p["x1"] * w,
                "y1": p["y1"] * h,
                "x2": p["x2"] * w,
                "y2": p["y2"] * h,
            })
    return out


def _normalize_results(results, w: int, h: int) -> list[dict]:
    """
    Convert mlx_sam raw results to normalized annotation format.

    Expected raw result per mask:
        {"segmentation": [[x, y], ...] or flat [x,y,...], "score": float}
    """
    output = []
    for r in results:
        raw_seg = r.get("segmentation", [])

        # Flatten if nested [[x,y], ...]
        if raw_seg and isinstance(raw_seg[0], (list, tuple)):
            flat = []
            for pt in raw_seg:
                flat.extend(pt)
            raw_seg = flat

        norm_seg = []
        for i, v in enumerate(raw_seg):
            norm_seg.append(v / w if i % 2 == 0 else v / h)

        # Derive bbox from segmentation
        if norm_seg:
            xs = norm_seg[0::2]
            ys = norm_seg[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        else:
            bbox = r.get("bbox", [])

        output.append({
            "segmentation": norm_seg,
            "bbox": bbox,
            "score": r.get("score", 0.0),
        })
    return output
