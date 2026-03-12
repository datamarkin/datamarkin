"""SAM API Blueprint - Minimal implementation with point and text prompts."""

from pathlib import Path

from flask import Blueprint, jsonify, request

from config import SAM_MODELS_DIR, file_path as get_file_path
from queries import get_file_by_id

sam3_api = Blueprint("sam_api", __name__, url_prefix="/api/sam")


def _ok(data, status=200):
    return jsonify({"data": data}), status


def _err(message, code, status):
    return jsonify({"error": {"message": message, "code": code}}), status


@sam3_api.route("/predict_points", methods=["POST"])
def sam_predict_points():
    """
    Point-based segmentation.

    Request:
      {
        "file_id": "...",
        "points": [[x1, y1], [x2, y2]],  # pixel coordinates
        "labels": [true, false]           # true=positive, false=negative
      }

    Response:
      {"data": {"masks": [{"segmentation": [...], "bbox": [...], "score": ...}, ...]}}
    """
    from PIL import Image as PILImage

    body = request.get_json(silent=True) or {}
    file_id = body.get("file_id")
    points = body.get("points", [])  # [[x, y], ...] pixel coords
    labels = body.get("labels", [])  # [true/false, ...]

    if not file_id:
        return _err("'file_id' is required", "missing_fields", 400)
    if not points or len(points) != len(labels):
        return _err(
            "'points' and 'labels' must be non-empty lists of same length",
            "invalid_request",
            400,
        )

    # Look up file
    file_row = get_file_by_id(file_id)
    if not file_row:
        return _err("File not found", "not_found", 404)

    image_path = get_file_path(file_row["filename"])
    if not image_path.exists():
        return _err("Image file not found on disk", "file_missing", 404)

    # Load image to get dimensions
    try:
        with PILImage.open(image_path) as img:
            width, height = img.size
    except Exception as exc:
        return _err(f"Failed to read image: {exc}", "image_error", 500)

    # Get backend and encode image (or use cache)
    try:
        from sam3_backend import get_sam_backend

        backend = get_sam_backend()

        if not backend.is_available():
            return _err("SAM backend not available", "backend_unavailable", 503)

        # Load model if needed
        model_loaded = getattr(backend, "_model", None) is not None
        if not model_loaded:
            target_variant = _find_first_available_variant()
            if not target_variant:
                return (
                    _err("No SAM weights found. Download weights first.", "no_weights", 400)
                )
            backend.load(SAM_MODELS_DIR / target_variant)

        # Encode image (uses cache internally)
        embedding_id = backend.encode_image(image_path)

    except RuntimeError as exc:
        return _err(str(exc), "backend_error", 503)

    try:
        # Normalize points to [0, 1]
        norm_points = [[x / width, y / height] for x, y in points]
        bool_labels = [bool(l) for l in labels]

        # Run prediction
        masks = backend.predict(embedding_id, norm_points, bool_labels, width, height)
    except KeyError:
        return _err("Failed to encode image", "encode_error", 500)
    except Exception as exc:
        return _err(f"Prediction failed: {exc}", "predict_failed", 500)

    return _ok({"masks": masks})


@sam3_api.route("/predict_text", methods=["POST"])
def sam_predict_text():
    """
    Text-based segmentation.

    Request:
      {
        "file_id": "...",
        "text_prompt": "car",
        "confidence_threshold": 0.5
      }

    Response:
      {"data": {"masks": [{"segmentation": [...], "bbox": [...], "score": ...}, ...]}}
    """
    from PIL import Image as PILImage

    body = request.get_json(silent=True) or {}
    file_id = body.get("file_id")
    text_prompt = body.get("text_prompt", "").strip()
    confidence_threshold = body.get("confidence_threshold", 0.5)

    if not file_id:
        return _err("'file_id' is required", "missing_fields", 400)
    if not text_prompt:
        return _err("'text_prompt' is required", "missing_fields", 400)

    # Look up file
    file_row = get_file_by_id(file_id)
    if not file_row:
        return _err("File not found", "not_found", 404)

    image_path = get_file_path(file_row["filename"])
    if not image_path.exists():
        return _err("Image file not found on disk", "file_missing", 404)

    # Load image to get dimensions
    try:
        with PILImage.open(image_path) as img:
            width, height = img.size
    except Exception as exc:
        return _err(f"Failed to read image: {exc}", "image_error", 500)

    # Get backend and encode image (or use cache)
    try:
        from sam3_backend import get_sam_backend

        backend = get_sam_backend()

        if not backend.is_available():
            return _err("SAM backend not available", "backend_unavailable", 503)

        # Load model if needed
        model_loaded = getattr(backend, "_model", None) is not None
        if not model_loaded:
            target_variant = _find_first_available_variant()
            if not target_variant:
                return (
                    _err("No SAM weights found. Download weights first.", "no_weights", 400)
                )
            backend.load(SAM_MODELS_DIR / target_variant)

        # Encode image (uses cache internally)
        embedding_id = backend.encode_image(image_path)

    except RuntimeError as exc:
        return _err(str(exc), "backend_error", 503)

    try:
        # Run prediction with text prompt
        masks = backend.predict_text(
            embedding_id, text_prompt, width, height, float(confidence_threshold)
        )
    except KeyError:
        return _err("Failed to encode image", "encode_error", 500)
    except Exception as exc:
        return _err(f"Prediction failed: {exc}", "predict_failed", 500)

    return _ok({"masks": masks})


@sam3_api.route("/unload", methods=["POST"])
def sam_unload():
    try:
        from sam3_backend import get_sam_backend

        backend = get_sam_backend()
        backend.unload()
    except Exception as exc:
        return _err(str(exc), "unload_failed", 500)

    return _ok({"unloaded": True})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_first_available_variant() -> str | None:
    if not SAM_MODELS_DIR.exists():
        return None
    for variant_dir in sorted(SAM_MODELS_DIR.iterdir()):
        if variant_dir.is_dir():
            has_weights = any(
                f.suffix in (".safetensors", ".pt", ".pth", ".bin")
                for f in variant_dir.iterdir()
            )
            if has_weights:
                return variant_dir.name
    return None
