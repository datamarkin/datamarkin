"""Falcon-Perception API — text-prompted detection & segmentation."""

import gc
import json
import sys
import threading
import time
import urllib.request

import pixelflow as pf
from flask import Blueprint, jsonify, request
from PIL import Image as PILImage
from pixelflow.assets import DownloadError

import task_queue
from config import DATA_DIR, file_path as get_file_path
from queries import get_file_by_id, get_project_by_id, get_project_files, update_file_annotations, update_file_analyzed_labels
from routes.download_api import update_download_state, clear_download_state
from routes.predict_route import mask_to_norm_polygon
from utils.dedup import deduplicate_objects

_USE_MLX = sys.platform == "darwin"

if _USE_MLX:
    import mlx.core as mx

falcon_perception_api = Blueprint("falcon_perception_api", __name__, url_prefix="/api/falcon")

_engine = None
_tokenizer = None
_model_args = None
_lock = threading.Lock()


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

    if _USE_MLX:
        from falcon_perception.mlx.batch_inference import BatchInferenceEngine
        model, _tokenizer, _model_args = load_and_prepare_model(
            hf_local_dir=str(_FALCON_LOCAL), dtype="float16", backend="mlx",
        )
    else:
        from falcon_perception.batch_inference import BatchInferenceEngine
        model, _tokenizer, _model_args = load_and_prepare_model(
            hf_local_dir=str(_FALCON_LOCAL), dtype="float16", backend="torch",
        )

    _engine = BatchInferenceEngine(model, _tokenizer)


def _clear_gpu_cache():
    """Clear GPU memory cache for the active backend."""
    if _USE_MLX:
        mx.metal.clear_cache()
    else:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _predict_single(pil_image, query, task="segmentation", max_new_tokens=1014):
    """Run falcon-perception for one query on one image. Returns pf Detections."""
    from falcon_perception import build_prompt_for_task

    if _USE_MLX:
        from falcon_perception.mlx.batch_inference import process_batch_and_generate
    else:
        from falcon_perception.batch_inference import process_batch_and_generate

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
    _clear_gpu_cache()

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


def _auto_annotate_file(file_row, labels, force=False):
    """Run falcon-perception on one file for un-analyzed labels.

    Returns (new_objects, all_objects, analyzed_label_names).
    """
    existing = _get_existing_objects(file_row)
    label_name_to_id = {l["name"].lower(): l["id"] for l in labels}

    # Determine which labels still need analysis
    already_analyzed = set()
    if not force:
        raw_al = file_row.get("analyzed_labels")
        if raw_al:
            try:
                already_analyzed = set(json.loads(raw_al) if isinstance(raw_al, str) else raw_al)
            except (json.JSONDecodeError, TypeError):
                already_analyzed = set()

    labels_to_run = [l for l in labels if l["name"].lower() not in already_analyzed]

    if not labels_to_run:
        return [], existing, sorted(already_analyzed)

    image_path = get_file_path(file_row["filename"])
    pil_image = PILImage.open(image_path).convert("RGB")
    img_w, img_h = pil_image.size

    all_new = []
    for label in labels_to_run:
        detections = _predict_single(pil_image, label["name"])
        objects = _detections_to_norm_objects(detections, label_name_to_id, img_w, img_h)
        all_new.extend(objects)
        del detections

    gc.collect()
    _clear_gpu_cache()

    new_objects = deduplicate_objects(all_new, existing)
    merged = existing + new_objects
    newly_analyzed = already_analyzed | {l["name"].lower() for l in labels_to_run}
    return new_objects, merged, sorted(newly_analyzed)


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
    force = body.get("force", False)

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
            new_objects, merged, analyzed = _auto_annotate_file(file_row, labels, force=force)
    except DownloadError as e:
        return jsonify({"error": f"Model download failed: {e}"}), 503

    update_file_annotations(file_id, json.dumps({"objects": merged}))
    update_file_analyzed_labels(file_id, json.dumps(analyzed))
    return jsonify({"new_count": len(new_objects), "objects": new_objects})


def _auto_annotate_executor(ctx):
    """Task queue executor for batch auto-annotation."""
    files = ctx.meta["files"]
    labels = ctx.meta["labels"]
    force = ctx.meta.get("force", False)

    with _lock:
        _ensure_loaded()
        for i, file_row in enumerate(files):
            if ctx.is_cancelled():
                return
            new_objects, merged, analyzed = _auto_annotate_file(
                file_row, labels, force=force,
            )
            if new_objects:
                update_file_annotations(file_row["id"], json.dumps({"objects": merged}))
            update_file_analyzed_labels(file_row["id"], json.dumps(analyzed))
            ctx.progress((i + 1) / len(files), f"{i + 1}/{len(files)} images")


@falcon_perception_api.route("/auto_annotate_batch", methods=["POST"])
def falcon_auto_annotate_batch():
    """Start batch auto-annotation for all project images."""
    body = request.get_json(silent=True) or {}
    project_id = body.get("project_id")
    target = body.get("target", "all")
    force = body.get("force", False)

    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    project = get_project_by_id(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    labels = _parse_labels(project)
    if not labels:
        return jsonify({"error": "Project has no labels"}), 400

    files = get_project_files(project_id)
    if target == "pending":
        files = [f for f in files if not f.get("annotations") or f["annotations"] in ("", "[]", "null")]

    # Skip files where all current labels have already been analyzed
    if not force:
        label_names_set = {l["name"].lower() for l in labels}
        def _needs_analysis(f):
            raw = f.get("analyzed_labels")
            if not raw or raw in ("", "[]", "null"):
                return True
            try:
                already = set(json.loads(raw) if isinstance(raw, str) else raw)
            except (json.JSONDecodeError, TypeError):
                return True
            return not label_names_set.issubset(already)
        files = [f for f in files if _needs_analysis(f)]

    if not files:
        return jsonify({"status": "done", "total": 0})

    task_id = task_queue.submit(
        "auto_annotate",
        _auto_annotate_executor,
        label=f"Auto-annotating {project['name']}",
        meta={"project_id": project_id, "force": force,
              "files": files, "labels": labels},
    )
    return jsonify({"status": "started", "total": len(files), "task_id": task_id})


@falcon_perception_api.route("/auto_annotate_batch_status", methods=["GET"])
def falcon_auto_annotate_batch_status():
    """Backward-compatible status endpoint for auto-annotate.js polling."""
    task = task_queue.find_recent_task("auto_annotate")
    if not task:
        return jsonify({"status": "idle", "current": 0, "total": 0, "file_id": None, "error": None})
    total = len(task["meta"].get("files", []))
    current = int(task["progress"] * total)
    status = task["status"]
    if status in ("cancelled", "done"):
        status = "done"
    elif status == "failed":
        status = "error"
    return jsonify({
        "status": status,
        "current": current,
        "total": total,
        "file_id": None,
        "error": task.get("error"),
    })
