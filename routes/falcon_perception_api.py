"""Falcon-Perception API — text-prompted detection & segmentation."""

import gc
import json
import threading
import time
import urllib.request

import mlx.core as mx
import pixelflow as pf
from flask import Blueprint, jsonify, request
from PIL import Image as PILImage
from pixelflow.assets import DownloadError

from config import DATA_DIR, file_path as get_file_path
from queries import get_file_by_id, get_project_by_id, get_project_files, update_file_annotations
from routes.download_api import update_download_state, clear_download_state
from routes.predict_route import mask_to_norm_polygon
from utils.dedup import deduplicate_objects

falcon_perception_api = Blueprint("falcon_perception_api", __name__, url_prefix="/api/falcon")

_engine = None
_tokenizer = None
_model_args = None
_lock = threading.Lock()

_batch_state = {"status": "idle", "current": 0, "total": 0, "file_id": None, "error": None}
_batch_lock = threading.Lock()


def _parse_labels(project):
    raw = project.get("labels", "[]")
    return json.loads(raw) if isinstance(raw, str) else raw


_FALCON_FILES = [
    "config.json", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json",
]
_FALCON_LOCAL = DATA_DIR / "dtmfiles" / "falcon-perception"


_WEIGHTS_URL = "https://dtmfiles.com/falcon-perception/model.safetensors"
_WEIGHTS_LOCAL = _FALCON_LOCAL / "model.safetensors"


def _download_falcon_files():
    """Download all Falcon Perception files with progress tracking for the weights."""
    # Download small config/tokenizer files first (instant if cached)
    for f in _FALCON_FILES:
        if f == "model.safetensors":
            continue
        pf.assets.download(f"falcon-perception/{f}", directory=DATA_DIR)

    # If weights already cached, skip
    if _WEIGHTS_LOCAL.exists():
        return

    # Get Content-Length for progress tracking
    total = 0
    try:
        req = urllib.request.Request(_WEIGHTS_URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as r:
            total = int(r.headers.get("Content-Length", 0))
    except Exception:
        pass

    update_download_state("Falcon Perception", status="downloading", pct=0)
    tmp = _WEIGHTS_LOCAL.with_suffix(".safetensors.download")
    exc = [None]

    def do_download():
        try:
            pf.assets.download("falcon-perception/model.safetensors", directory=DATA_DIR, quiet=True)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=do_download, daemon=True)
    t.start()
    while t.is_alive():
        if total > 0 and tmp.exists():
            try:
                update_download_state("Falcon Perception", status="downloading",
                                      pct=min(99, int(tmp.stat().st_size / total * 100)))
            except OSError:
                pass
        time.sleep(0.5)
    t.join()

    if exc[0] is not None:
        update_download_state("Falcon Perception", status="error", error=str(exc[0]))
        raise exc[0]

    update_download_state("Falcon Perception", status="ready", pct=100)
    # Clear after a delay so the toast can show "Complete"
    threading.Timer(5.0, clear_download_state, args=["Falcon Perception"]).start()


def _ensure_loaded():
    global _engine, _tokenizer, _model_args
    if _engine is not None:
        return

    _download_falcon_files()

    from falcon_perception import load_and_prepare_model
    from falcon_perception.mlx.batch_inference import BatchInferenceEngine

    model, _tokenizer, _model_args = load_and_prepare_model(
        local_dir=str(_FALCON_LOCAL), dtype="float16", backend="mlx",
    )
    _engine = BatchInferenceEngine(model, _tokenizer)


def _predict_single(pil_image, query, task="segmentation", max_new_tokens=1014):
    """Run falcon-perception for one query on one image. Returns pf Detections."""
    from falcon_perception import build_prompt_for_task
    from falcon_perception.mlx.batch_inference import process_batch_and_generate

    prompt = build_prompt_for_task(query, task)
    batch = process_batch_and_generate(
        _tokenizer, [(pil_image, prompt)],
        max_length=_model_args.max_seq_len, min_dimension=256, max_dimension=1024,
    )
    _, aux_outputs = _engine.generate(
        tokens=batch["tokens"], pos_t=batch["pos_t"], pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
        max_new_tokens=max_new_tokens, temperature=0.0, task=task,
    )
    aux = aux_outputs[0]
    w, h = pil_image.size
    detections = pf.detections.from_falcon_perception(aux, image_size=(w, h), label=query)

    del aux_outputs, aux, batch
    mx.metal.clear_cache()

    return detections


def _detections_to_norm_objects(detections, label_name_to_id, img_w, img_h):
    """Convert pixelflow Detections to normalized annotation objects.

    Maps det.class_name (query text) to the matching project label id.
    Returns list of {class, bbox, segmentation?} dicts.
    """
    objects = []
    for det in detections:
        class_name = (det.class_name or "").lower()
        label_id = label_name_to_id.get(class_name)
        if label_id is None:
            continue

        x1, y1, x2, y2 = det.bbox
        norm_bbox = [
            float(x1) / img_w,
            float(y1) / img_h,
            float(x2) / img_w,
            float(y2) / img_h,
        ]

        obj = {"class": label_id, "bbox": norm_bbox}

        if det.masks:
            for mask in det.masks:
                poly = mask_to_norm_polygon(mask, img_w, img_h)
                if poly:
                    obj["segmentation"] = poly
                    break

        objects.append(obj)
    return objects


def _get_existing_objects(file_row):
    """Parse existing annotation objects from a file row."""
    raw = file_row.get("annotations")
    if not raw or raw in ("", "[]", "null"):
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return data.get("objects", []) if isinstance(data, dict) else []
    except (json.JSONDecodeError, AttributeError):
        return []


def _auto_annotate_file(file_row, labels):
    """Run falcon-perception on one file for all labels. Returns (new_objects, all_objects)."""
    image_path = get_file_path(file_row["filename"])
    pil_image = PILImage.open(image_path).convert("RGB")
    img_w, img_h = pil_image.size

    existing = _get_existing_objects(file_row)
    label_name_to_id = {l["name"].lower(): l["id"] for l in labels}

    all_new = []
    for label in labels:
        detections = _predict_single(pil_image, label["name"])
        objects = _detections_to_norm_objects(detections, label_name_to_id, img_w, img_h)
        all_new.extend(objects)
        del detections

    gc.collect()
    mx.metal.clear_cache()

    new_objects = deduplicate_objects(all_new, existing)
    merged = existing + new_objects
    return new_objects, merged


# ── Endpoints ────────────────────────────────────────────────────────────────


@falcon_perception_api.route("/load", methods=["POST"])
def falcon_load():
    try:
        with _lock:
            _ensure_loaded()
    except DownloadError as e:
        return jsonify({"error": f"Model download failed: {e}"}), 503
    return jsonify({"data": {"loaded": True}})


@falcon_perception_api.route("/auto_annotate", methods=["POST"])
def falcon_auto_annotate():
    """Auto-annotate a single image with all project labels."""
    body = request.get_json(silent=True) or {}
    file_id = body.get("file_id")
    project_id = body.get("project_id")

    if not file_id or not project_id:
        return jsonify({"error": "file_id and project_id are required"}), 400

    project = get_project_by_id(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    labels = _parse_labels(project)
    if not labels:
        return jsonify({"error": "Project has no labels"}), 400

    file_row = get_file_by_id(file_id)
    if not file_row:
        return jsonify({"error": "File not found"}), 404

    try:
        with _lock:
            _ensure_loaded()
            new_objects, merged = _auto_annotate_file(file_row, labels)
    except DownloadError as e:
        return jsonify({"error": f"Model download failed: {e}"}), 503

    update_file_annotations(file_id, json.dumps({"objects": merged}))
    return jsonify({"new_count": len(new_objects), "objects": new_objects})


@falcon_perception_api.route("/auto_annotate_batch", methods=["POST"])
def falcon_auto_annotate_batch():
    """Start batch auto-annotation for all project images."""
    global _batch_state
    body = request.get_json(silent=True) or {}
    project_id = body.get("project_id")
    target = body.get("target", "all")

    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    with _batch_lock:
        if _batch_state["status"] == "running":
            return jsonify({"error": "Batch already running"}), 409
        _batch_state = {"status": "running", "current": 0, "total": 0, "file_id": None, "error": None}

    project = get_project_by_id(project_id)
    if not project:
        with _batch_lock:
            _batch_state = {"status": "error", "current": 0, "total": 0, "file_id": None, "error": "Project not found"}
        return jsonify({"error": "Project not found"}), 404

    labels = _parse_labels(project)
    if not labels:
        with _batch_lock:
            _batch_state = {"status": "error", "current": 0, "total": 0, "file_id": None, "error": "No labels"}
        return jsonify({"error": "Project has no labels"}), 400

    files = get_project_files(project_id)
    if target == "pending":
        files = [f for f in files if not f.get("annotations") or f["annotations"] in ("", "[]", "null")]

    with _batch_lock:
        _batch_state["total"] = len(files)

    def run_batch():
        global _batch_state
        try:
            with _lock:
                _ensure_loaded()
            for i, file_row in enumerate(files):
                with _batch_lock:
                    _batch_state["current"] = i + 1
                    _batch_state["file_id"] = file_row["id"]

                with _lock:
                    new_objects, merged = _auto_annotate_file(file_row, labels)
                if new_objects:
                    update_file_annotations(file_row["id"], json.dumps({"objects": merged}))

            with _batch_lock:
                _batch_state["status"] = "done"
        except Exception as e:
            with _batch_lock:
                _batch_state["status"] = "error"
                _batch_state["error"] = str(e)

    t = threading.Thread(target=run_batch, daemon=True)
    t.start()
    return jsonify({"status": "started", "total": len(files)})


@falcon_perception_api.route("/auto_annotate_batch_status", methods=["GET"])
def falcon_auto_annotate_batch_status():
    with _batch_lock:
        return jsonify(dict(_batch_state))
