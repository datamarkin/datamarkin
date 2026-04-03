"""EfficientTAM API (PyTorch + MPS backend)."""

import gc
import threading
import time
import urllib.request

import numpy as np
import pixelflow as pf
import torch
from flask import Blueprint, jsonify, request
from PIL import Image as PILImage

from config import DATA_DIR, EFFICIENTTAM_MODELS_DIR, file_path as get_file_path
from queries import get_file_by_id
from routes.download_api import update_download_state, clear_download_state
from routes.predict_route import mask_to_norm_polygon

efficienttam_api = Blueprint("sam_api", __name__, url_prefix="/api/sam")

_predictor = None
_cached_file_id = None
_lock = threading.Lock()

_dl_state = {"status": "idle", "pct": 0, "error": None}
_dl_lock = threading.Lock()

_MODEL_ASSET = "EfficientTAM/efficienttam_s.pt"
_MODEL_URL = "https://dtmfiles.com/EfficientTAM/efficienttam_s.pt"


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _model_path():
    return EFFICIENTTAM_MODELS_DIR / "efficienttam_s.pt"


def _ensure_loaded():
    global _predictor, _cached_file_id
    if _predictor is not None:
        return
    ckpt = _model_path()
    if not ckpt.exists():
        raise FileNotFoundError("SAM model not downloaded. Use /api/sam/download_model first.")
    from efficient_track_anything.build_efficienttam import build_efficienttam
    from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor

    model = build_efficienttam(
        "configs/efficienttam/efficienttam_s.yaml",
        str(ckpt),
        device=_device(),
        hydra_overrides_extra=["++model.compile_image_encoder=false"],
    )
    _predictor = EfficientTAMImagePredictor(model)
    _cached_file_id = None


def _ensure_embedding(file_id):
    global _cached_file_id
    if _cached_file_id == file_id:
        return
    file_row = get_file_by_id(file_id)
    if not file_row:
        raise ValueError(f"File not found: {file_id}")
    image_path = get_file_path(file_row["filename"])
    image = np.array(PILImage.open(image_path).convert("RGB"))
    _predictor.set_image(image)
    _cached_file_id = file_id




def _run_download():
    global _dl_state

    with _dl_lock:
        if _dl_state["status"] == "downloading":
            return
        _dl_state.update({"status": "downloading", "pct": 0, "error": None})

    update_download_state("EfficientTAM", status="downloading", pct=0)

    try:
        # Pre-fetch Content-Length for progress tracking
        total = 0
        try:
            req = urllib.request.Request(_MODEL_URL, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as r:
                total = int(r.headers.get("Content-Length", 0))
        except Exception:
            pass

        tmp = _model_path().with_suffix(".pt.download")
        exc = [None]

        def do_download():
            try:
                pf.assets.download(_MODEL_ASSET, directory=DATA_DIR, quiet=True)
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=do_download, daemon=True)
        t.start()

        while t.is_alive():
            if total > 0 and tmp.exists():
                try:
                    pct = min(99, int(tmp.stat().st_size / total * 100))
                    _dl_state["pct"] = pct
                    update_download_state("EfficientTAM", status="downloading", pct=pct)
                except OSError:
                    pass
            time.sleep(0.5)
        t.join()

        if exc[0] is not None:
            raise exc[0]

        _dl_state.update({"status": "ready", "pct": 100, "error": None})
        update_download_state("EfficientTAM", status="ready", pct=100)
        threading.Timer(5.0, clear_download_state, args=["EfficientTAM"]).start()
    except Exception as e:
        _dl_state.update({"status": "error", "error": str(e)})
        update_download_state("EfficientTAM", status="error", error=str(e))


def _get_model_status():
    if _model_path().exists():
        return {"status": "ready", "pct": 100, "error": None}
    return dict(_dl_state)


@efficienttam_api.route("/model_status", methods=["GET"])
def sam_model_status():
    return jsonify({"data": _get_model_status()})


@efficienttam_api.route("/download_model", methods=["POST"])
def sam_download_model():
    status = _get_model_status()
    if status["status"] == "ready":
        return jsonify({"data": status})
    if _dl_state["status"] == "downloading":
        return jsonify({"data": _dl_state})
    t = threading.Thread(target=_run_download, daemon=True)
    t.start()
    return jsonify({"data": {"status": "downloading", "pct": 0, "error": None}})


@efficienttam_api.route("/load", methods=["POST"])
def sam_load():
    _ensure_loaded()
    return jsonify({"data": {"loaded": True}})


@efficienttam_api.route("/create_embedding", methods=["POST"])
def sam_create_embedding():
    body = request.get_json(silent=True) or {}
    file_id = body.get("file_id")
    with _lock:
        _ensure_loaded()
        _ensure_embedding(file_id)
    return jsonify({"data": {"file_id": file_id, "cached": True}})


@efficienttam_api.route("/predict_points", methods=["POST"])
def sam_predict_points():
    body = request.get_json(silent=True) or {}
    file_id = body.get("file_id")
    points = body.get("points", [])
    labels = body.get("labels", [])

    file_row = get_file_by_id(file_id)
    width = file_row["width"]
    height = file_row["height"]

    with _lock:
        _ensure_loaded()
        _ensure_embedding(file_id)

        coords = np.array(points, dtype=np.float32)   # [[px, py], ...]
        lbls = np.array([int(bool(l)) for l in labels], dtype=np.int32)

        masks, scores, _ = _predictor.predict(
            point_coords=coords,
            point_labels=lbls,
            multimask_output=True,
        )

        detections = pf.detections.from_efficienttam(masks, scores)

    gc.collect()

    if len(detections) == 0:
        return jsonify({"data": {"masks": []}})

    best = max(detections, key=lambda d: d.confidence)

    poly = mask_to_norm_polygon(best.masks[0], width, height)
    if not poly:
        return jsonify({"data": {"masks": []}})

    x1, y1, x2, y2 = best.bbox
    bbox = [x1 / width, y1 / height, x2 / width, y2 / height]

    return jsonify({"data": {"masks": [{"segmentation": poly, "bbox": bbox, "score": float(best.confidence)}]}})
