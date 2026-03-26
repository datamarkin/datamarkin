"""EfficientTAM API (PyTorch + MPS backend)."""

import gc
import threading
import time
import urllib.request

import cv2
import numpy as np
import torch
from flask import Blueprint, jsonify, request
from PIL import Image as PILImage

from config import DATA_DIR, EFFICIENTTAM_MODELS_DIR, file_path as get_file_path
from queries import get_file_by_id

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
    global _predictor
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


def _mask_to_polygon(mask_2d):
    mask_uint8 = (mask_2d > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    return max(contours, key=cv2.contourArea).reshape(-1).tolist()


def _run_download():
    global _dl_state
    import pixelflow as pf

    with _dl_lock:
        if _dl_state["status"] == "downloading":
            return
        _dl_state.update({"status": "downloading", "pct": 0, "error": None})

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
        result = [None]
        exc = [None]

        def do_download():
            try:
                result[0] = pf.assets.download(_MODEL_ASSET, directory=DATA_DIR, quiet=True)
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=do_download, daemon=True)
        t.start()

        while t.is_alive():
            if total > 0 and tmp.exists():
                try:
                    _dl_state["pct"] = min(99, int(tmp.stat().st_size / total * 100))
                except OSError:
                    pass
            time.sleep(0.5)
        t.join()

        if exc[0] is not None:
            raise exc[0]

        _dl_state.update({"status": "ready", "pct": 100, "error": None})
    except Exception as e:
        _dl_state.update({"status": "error", "error": str(e)})


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
    _ensure_loaded()
    with _lock:
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

    _ensure_loaded()

    with _lock:
        _ensure_embedding(file_id)

        coords = np.array(points, dtype=np.float32)   # [[px, py], ...]
        lbls = np.array([int(bool(l)) for l in labels], dtype=np.int32)

        masks, scores, _ = _predictor.predict(
            point_coords=coords,
            point_labels=lbls,
            multimask_output=True,
        )

        best = int(np.argmax(scores))
        polygon_px = _mask_to_polygon(masks[best])

    gc.collect()

    if not polygon_px:
        return jsonify({"data": {"masks": []}})

    norm_polygon = [
        polygon_px[i] / (width if i % 2 == 0 else height)
        for i in range(len(polygon_px))
    ]
    xs = norm_polygon[0::2]
    ys = norm_polygon[1::2]
    bbox = [min(xs), min(ys), max(xs), max(ys)]

    return jsonify({"data": {"masks": [{"segmentation": norm_polygon, "bbox": bbox, "score": float(scores[best])}]}})
