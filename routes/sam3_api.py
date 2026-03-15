"""SAM3 API — stripped to bare minimum to isolate memory leak."""

import gc
import threading
import cv2
import mlx.core as mx
import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image as PILImage

from config import file_path as get_file_path
from queries import get_file_by_id

sam3_api = Blueprint("sam_api", __name__, url_prefix="/api/sam")

# ---------------------------------------------------------------------------
# Globals — one model, one processor, one cached state
# ---------------------------------------------------------------------------
_processor = None
_cached_file_id = None
_cached_state = None
_lock = threading.Lock()


def _log_mem(label):
    """Log current Metal GPU memory usage."""
    active = mx.metal.get_active_memory() / (1024**2)
    cache = mx.metal.get_cache_memory() / (1024**2)
    print(f"[MEM] {label}: active={active:.0f}MB cache={cache:.0f}MB")


def _ensure_loaded():
    """Load model + processor on first call."""
    global _processor
    if _processor is not None:
        return

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    _log_mem("before model load")
    model = build_sam3_image_model()
    _processor = Sam3Processor(model, confidence_threshold=0.5)
    _log_mem("after model load")


def _ensure_embedding(file_id):
    """Cache embedding for exactly one image."""
    global _cached_file_id, _cached_state

    if _cached_file_id == file_id and _cached_state is not None:
        return _cached_state

    file_row = get_file_by_id(file_id)
    if not file_row:
        raise ValueError(f"File not found: {file_id}")

    image_path = get_file_path(file_row["filename"])
    image = PILImage.open(image_path).convert("RGB")

    _log_mem("before set_image")
    _cached_state = _processor.set_image(image)
    _cached_file_id = file_id
    _log_mem("after set_image")

    return _cached_state


def _reset(state):
    """Clean ALL inference artifacts from state. Only backbone_out +
    original_height + original_width should survive."""
    if "backbone_out" in state:
        for k in ("language_features", "language_mask", "language_embeds"):
            state["backbone_out"].pop(k, None)
    for k in ("geometric_prompt", "boxes", "masks", "scores"):
        state.pop(k, None)


def _mask_to_polygon(mask_2d):
    mask_uint8 = (mask_2d > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1).tolist()


def _format_results(state, width, height):
    masks = state.get("masks")
    boxes = state.get("boxes")
    scores = state.get("scores")
    if masks is None:
        return []

    # Force MLX evaluation and convert to numpy BEFORE we do anything else
    masks_np = np.array(masks)
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)

    results = []
    for i in range(masks_np.shape[0]):
        polygon_px = _mask_to_polygon(masks_np[i, 0])
        if not polygon_px:
            continue
        norm_polygon = []
        for j in range(0, len(polygon_px), 2):
            norm_polygon.append(polygon_px[j] / width)
            norm_polygon.append(polygon_px[j + 1] / height)
        box = boxes_np[i]
        results.append({
            "segmentation": norm_polygon,
            "bbox": [float(box[0]) / width, float(box[1]) / height,
                     float(box[2]) / width, float(box[3]) / height],
            "score": float(scores_np[i]),
        })
    return results


# ---------------------------------------------------------------------------
# Single route — predict_points
# ---------------------------------------------------------------------------

@sam3_api.route("/load", methods=["POST"])
def sam_load():
    _ensure_loaded()
    return jsonify({"data": {"loaded": True}})


@sam3_api.route("/create_embedding", methods=["POST"])
def sam_create_embedding():
    body = request.get_json(silent=True) or {}
    file_id = body.get("file_id")
    _ensure_loaded()
    _ensure_embedding(file_id)
    return jsonify({"data": {"file_id": file_id, "cached": True}})


@sam3_api.route("/predict_points", methods=["POST"])
def sam_predict_points():
    body = request.get_json(silent=True) or {}
    file_id = body.get("file_id")
    points = body.get("points", [])
    print(points)
    labels = body.get("labels", [])

    file_row = get_file_by_id(file_id)
    width = file_row["width"]
    height = file_row["height"]

    _ensure_loaded()

    with _lock:
        state = _ensure_embedding(file_id)

        _log_mem("before inference")

        # 1. Reset — exact same pattern as points_example.py
        _reset(state)

        # 2. Normalize and run inference (processor evals results internally)
        norm_points = [[px / width, py / height] for px, py in points]
        bool_labels = [bool(l) for l in labels]
        _processor.add_points_prompt(points=norm_points, labels=bool_labels, state=state)

        # 3. Format results (converts to pure python — no MLX references)
        masks = _format_results(state, width, height)

        # 4. Clean up inference tensors and release GPU memory to OS
        _reset(state)
        mx.clear_cache()
        gc.collect()

        _log_mem("after cleanup")

    return jsonify({"data": {"masks": masks}})


@sam3_api.route("/predict_text", methods=["POST"])
def sam_predict_text():
    body = request.get_json(silent=True) or {}
    file_id = body.get("file_id")
    text_prompt = (body.get("text_prompt") or "").strip()
    confidence_threshold = body.get("confidence_threshold")

    file_row = get_file_by_id(file_id)
    width = file_row["width"]
    height = file_row["height"]

    _ensure_loaded()

    with _lock:
        state = _ensure_embedding(file_id)

        _reset(state)

        if confidence_threshold is not None:
            _processor.confidence_threshold = float(confidence_threshold)

        _processor.set_text_prompt(prompt=text_prompt, state=state)
        masks = _format_results(state, width, height)
        _reset(state)
        mx.clear_cache()
        gc.collect()

    return jsonify({"data": {"masks": masks}})
