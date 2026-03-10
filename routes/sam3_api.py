"""
SAM API Blueprint

Endpoints:
    GET  /api/sam/status
    POST /api/sam/encode
    POST /api/sam/predict
    POST /api/sam/download
    GET  /api/sam/download/<variant>
    POST /api/sam/unload
"""

from pathlib import Path

from flask import Blueprint, jsonify, request

from config import SAM_MODELS_DIR, file_path as get_file_path
from queries import get_file_by_id

sam3_api = Blueprint("sam_api", __name__, url_prefix="/api/sam")


def _ok(data, status=200):
    return jsonify({"data": data}), status


def _err(message, code, status):
    return jsonify({"error": {"message": message, "code": code}}), status


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@sam3_api.route("/status", methods=["GET"])
def sam_status():
    from sam3_backend.status import get_sam_status
    status = get_sam_status(SAM_MODELS_DIR)
    return _ok(status)


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

@sam3_api.route("/encode", methods=["POST"])
def sam_encode():
    body = request.get_json(silent=True) or {}
    file_id = body.get("file_id")
    variant = body.get("variant")

    if not file_id:
        return _err("'file_id' is required", "missing_fields", 400)

    file_row = get_file_by_id(file_id)
    if not file_row:
        return _err("File not found", "not_found", 404)

    image_path = get_file_path(file_row["filename"])
    if not image_path.exists():
        return _err("Image file not found on disk", "file_missing", 404)

    try:
        backend = _get_loaded_backend(variant)
    except RuntimeError as exc:
        return _err(str(exc), "backend_unavailable", 503)

    try:
        embedding_id = backend.encode_image(image_path)
    except Exception as exc:
        return _err(str(exc), "encode_failed", 500)

    return _ok({"embedding_id": embedding_id})


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@sam3_api.route("/predict", methods=["POST"])
def sam_predict():
    body = request.get_json(silent=True) or {}
    embedding_id = body.get("embedding_id")
    prompts = body.get("prompts", [])
    width = body.get("width")
    height = body.get("height")

    if not embedding_id:
        return _err("'embedding_id' is required", "missing_fields", 400)
    if width is None or height is None:
        return _err("'width' and 'height' are required", "missing_fields", 400)

    try:
        from sam3_backend import get_sam_backend
        backend = get_sam_backend()
    except Exception as exc:
        return _err(str(exc), "backend_error", 500)

    if not backend.is_available():
        return _err("SAM backend not available", "backend_unavailable", 503)

    try:
        masks = backend.predict(embedding_id, prompts, int(width), int(height))
    except KeyError as exc:
        return _err(str(exc), "embedding_not_found", 404)
    except Exception as exc:
        return _err(str(exc), "predict_failed", 500)

    return _ok({"masks": masks})


# ---------------------------------------------------------------------------
# Text predict (cached embedding)
# ---------------------------------------------------------------------------

@sam3_api.route("/text", methods=["POST"])
def sam_text():
    body = request.get_json(silent=True) or {}
    embedding_id = body.get("embedding_id")
    text_prompt = body.get("text_prompt")
    width = body.get("width")
    height = body.get("height")
    confidence_threshold = body.get("confidence_threshold", 0.3)

    if not embedding_id:
        return _err("'embedding_id' is required", "missing_fields", 400)
    if not text_prompt:
        return _err("'text_prompt' is required", "missing_fields", 400)
    if width is None or height is None:
        return _err("'width' and 'height' are required", "missing_fields", 400)

    try:
        from sam3_backend import get_sam_backend
        backend = get_sam_backend()
    except Exception as exc:
        return _err(str(exc), "backend_error", 500)

    if not backend.is_available():
        return _err("SAM backend not available", "backend_unavailable", 503)

    try:
        masks = backend.predict_text(
            embedding_id, text_prompt, int(width), int(height), float(confidence_threshold)
        )
    except KeyError as exc:
        return _err(str(exc), "embedding_not_found", 404)
    except Exception as exc:
        return _err(str(exc), "predict_failed", 500)

    return _ok({"masks": masks})


# ---------------------------------------------------------------------------
# Text batch (encode + multi-label detection in one call)
# ---------------------------------------------------------------------------

@sam3_api.route("/text_batch", methods=["POST"])
def sam_text_batch():
    body = request.get_json(silent=True) or {}
    file_id = body.get("file_id")
    text_prompts = body.get("text_prompts", [])
    confidence_threshold = body.get("confidence_threshold", 0.3)

    if not file_id:
        return _err("'file_id' is required", "missing_fields", 400)
    if not text_prompts:
        return _err("'text_prompts' must be a non-empty list", "missing_fields", 400)

    file_row = get_file_by_id(file_id)
    if not file_row:
        return _err("File not found", "not_found", 404)

    image_path = get_file_path(file_row["filename"])
    if not image_path.exists():
        return _err("Image file not found on disk", "file_missing", 404)

    try:
        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            width, height = img.size
    except Exception as exc:
        return _err(f"Failed to read image: {exc}", "image_error", 500)

    try:
        backend = _get_loaded_backend(None)
    except RuntimeError as exc:
        return _err(str(exc), "backend_unavailable", 503)

    try:
        embedding_id = backend.encode_image(image_path)
    except Exception as exc:
        return _err(str(exc), "encode_failed", 500)

    detections = []
    for prompt in text_prompts:
        try:
            masks = backend.predict_text(
                embedding_id, prompt, width, height, float(confidence_threshold)
            )
            detections.extend(masks)
        except Exception as exc:
            return _err(f"Text prediction failed for '{prompt}': {exc}", "predict_failed", 500)

    return _ok({"detections": detections})


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

@sam3_api.route("/download", methods=["POST"])
def start_download():
    body = request.get_json(silent=True) or {}
    variant = body.get("variant")
    url = body.get("url")

    if not variant:
        return _err("'variant' is required", "missing_fields", 400)

    from sam3_backend import downloader
    dest_dir = SAM_MODELS_DIR / variant

    try:
        downloader.start(variant, dest_dir, url=url)
    except ValueError as exc:
        return _err(str(exc), "unknown_variant", 400)

    return _ok({"variant": variant, "state": "downloading"})


@sam3_api.route("/download/<variant>", methods=["GET"])
def download_progress(variant: str):
    from sam3_backend import downloader
    state = downloader.progress(variant)
    return _ok(state)


# ---------------------------------------------------------------------------
# Unload
# ---------------------------------------------------------------------------

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

def _get_loaded_backend(variant: str | None):
    """Return the backend, loading weights if necessary."""
    from sam3_backend import get_sam_backend
    backend = get_sam_backend()

    if not backend.is_available():
        raise RuntimeError("SAM framework is not installed")

    # If no model is loaded yet, try to auto-load from the first available variant
    try:
        # A quick probe: try to use the backend — if model not loaded it will raise
        # We detect this by checking private attr (best effort)
        model_loaded = getattr(backend, "_model", None) is not None
    except Exception:
        model_loaded = False

    if not model_loaded:
        target_variant = variant or _find_first_available_variant()
        if target_variant is None:
            raise RuntimeError("No SAM weights found. Download weights first.")
        model_dir = SAM_MODELS_DIR / target_variant
        backend.load(model_dir)

    return backend


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
