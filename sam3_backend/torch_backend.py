"""
CUDA / PyTorch SAM2 backend.

Wraps the official Facebook sam2 library. This file is only imported on
non-darwin platforms; import errors are caught by the factory in sam/__init__.py.
"""

from pathlib import Path

# Existence check — ImportError propagates to factory → UnavailableBackend.
import torch  # noqa: F401
import sam2  # noqa: F401

from sam3_backend.base import SAMBackend


class TorchBackend(SAMBackend):
    def __init__(self) -> None:
        self._predictor = None
        self._embeddings: dict[str, object] = {}

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            import importlib.util
            return (
                importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("sam2") is not None
            )
        except Exception:
            return False

    # ------------------------------------------------------------------
    def load(self, model_dir: Path) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Locate checkpoint and config inside model_dir
        checkpoints = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")
        checkpoint = checkpoints[0]

        configs = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))
        if not configs:
            raise FileNotFoundError(f"No config file found in {model_dir}")
        config = configs[0]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2(str(config), str(checkpoint), device=device)
        self._predictor = SAM2ImagePredictor(model)
        self._embeddings.clear()

    # ------------------------------------------------------------------
    def encode_image(self, image_path: Path) -> str:
        if self._predictor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import numpy as np
        from PIL import Image

        embedding_id = image_path.stem
        if embedding_id not in self._embeddings:
            img = np.array(Image.open(image_path).convert("RGB"))
            self._predictor.set_image(img)
            # Cache the internal state after set_image
            self._embeddings[embedding_id] = self._predictor.get_image_embedding()

        return embedding_id

    # ------------------------------------------------------------------
    def predict(
        self,
        embedding_id: str,
        prompts: list[dict],
        width: int,
        height: int,
    ) -> list[dict]:
        if self._predictor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if embedding_id not in self._embeddings:
            raise KeyError(f"No cached embedding for '{embedding_id}'. Call encode_image() first.")

        import numpy as np

        # Rebuild predictor state from cached embedding
        self._predictor._features = self._embeddings[embedding_id]

        point_coords, point_labels, box = None, None, None

        for p in prompts:
            if p["type"] == "point":
                coord = np.array([[p["x"] * width, p["y"] * height]])
                label = np.array([p.get("polarity", 1)])
                if point_coords is None:
                    point_coords, point_labels = coord, label
                else:
                    point_coords = np.vstack([point_coords, coord])
                    point_labels = np.concatenate([point_labels, label])
            elif p["type"] == "box":
                box = np.array([
                    p["x1"] * width,
                    p["y1"] * height,
                    p["x2"] * width,
                    p["y2"] * height,
                ])

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        results = []
        for mask, score in zip(masks, scores):
            contour_pts = _mask_to_polygon(mask)
            norm_seg = []
            for i, v in enumerate(contour_pts):
                norm_seg.append(v / width if i % 2 == 0 else v / height)

            xs = norm_seg[0::2]
            ys = norm_seg[1::2]
            if xs and ys:
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                bbox = []

            results.append({
                "segmentation": norm_seg,
                "bbox": bbox,
                "score": float(score),
            })

        return results

    # ------------------------------------------------------------------
    def unload(self) -> None:
        self._predictor = None
        self._embeddings.clear()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mask_to_polygon(mask) -> list[float]:
    """Convert a boolean mask (H×W numpy array) to a flat polygon [x,y,...]."""
    import cv2
    import numpy as np

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2)
    flat = []
    for x, y in pts:
        flat.extend([float(x), float(y)])
    return flat
