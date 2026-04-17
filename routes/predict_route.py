import base64
import json
import os
from io import BytesIO

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import pixelflow as pf
from flask import Blueprint, jsonify, request
from PIL import Image

from pathlib import Path

from config import DATA_DIR, file_path
from queries import get_file_by_id, get_training
from mozo import ModelManager

predict_api = Blueprint("predict_api", __name__)

# Singleton model manager for lazy loading and memory management
model_manager = ModelManager()


def _resolve_model_path(model_path):
    """Return a local file path for model weights.

    If the path already exists on disk (e.g. RF-DETR trained checkpoints),
    return it as-is.  Otherwise treat it as a dtmfiles.com asset path and
    download it (cached after the first fetch).
    """
    local = Path(model_path)
    if local.exists():
        return str(local)
    return str(pf.assets.download(model_path, directory=DATA_DIR))


def load_model_from_training(training):
    """Load a model from a training record.

    Reads architecture, labels, variant, checkpoint path from the training's
    config and dispatches to the correct mozo adapter.  Resolves model_path
    locally or via dtmfiles download.
    """
    cfg = training["config"] if isinstance(training["config"], dict) else json.loads(training["config"])
    training_id = training["id"]
    architecture = cfg.get("model_architecture", "rfdetr")
    labels = cfg.get("labels", [])
    checkpoint = _resolve_model_path(training["model_path"])

    project_type = cfg.get("project_type", "detection")
    project_type = {"object_detection": "detection", "instance_segmentation": "segmentation"}.get(project_type, project_type)

    if architecture == "detectron2":
        # training_id is passed as the 'variant' positional arg (used as cache key by ModelManager).
        # When config_path is set, the adapter loads config from file and doesn't need the variant name.
        # When config_path is absent, variant is used for model zoo config lookup.
        variant = cfg.get("variant", "mask_rcnn_R_50_FPN_3x")
        kwargs = dict(
            checkpoint_path=checkpoint,
            labels=labels,
            device='cpu',
        )
        config_path = cfg.get("config_path")
        if config_path:
            kwargs["config_path"] = str(_resolve_model_path(config_path))
            return model_manager.get_model('detectron2', training_id, **kwargs)
        return model_manager.get_model('detectron2', variant, **kwargs)
    # Default: RF-DETR
    return model_manager.get_model(
        'rfdetr',
        training_id,
        checkpoint_path=checkpoint,
        model_size=cfg.get("model_size", "base"),
        project_type=project_type,
        resolution=cfg.get("resolution", 560),
        labels=labels,
    )


def mask_to_norm_polygon(mask: np.ndarray, img_w: int, img_h: int):
    """Convert a boolean H×W mask to a normalized flat polygon [x1,y1,x2,y2,...].
    Returns None if no valid contour is found."""
    uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None
    poly = []
    for x, y in pts:
        poly.append(float(x) / img_w)
        poly.append(float(y) / img_h)
    return poly


def detections_to_objects(detections, labels, img_w: int, img_h: int,
                           class_name_key: bool = False) -> list:
    """Convert pixelflow Detections to normalized app objects.

    If class_name_key is True, objects use 'class_name' (for inference display).
    Otherwise they use 'class' label id (for annotation storage).
    """
    objects = []
    if detections is None or len(detections) == 0:
        return objects

    for det in detections:
        class_idx = int(det.class_id)
        if class_idx >= len(labels):
            continue

        # bbox is [x1, y1, x2, y2] in pixels
        x1, y1, x2, y2 = det.bbox
        norm_bbox = [
            float(x1) / img_w,
            float(y1) / img_h,
            float(x2 - x1) / img_w,
            float(y2 - y1) / img_h,
        ]
        confidence = float(det.confidence) if det.confidence is not None else None

        if class_name_key:
            obj = {"class_name": labels[class_idx]["name"], "bbox": norm_bbox}
        else:
            obj = {"class": labels[class_idx]["id"], "bbox": norm_bbox}
        if confidence is not None:
            obj["confidence"] = round(confidence, 4)

        # Handle segmentation masks (det.masks is a list of boolean arrays)
        if det.masks:
            for mask in det.masks:
                poly = mask_to_norm_polygon(mask, img_w, img_h)
                if poly:
                    obj["segmentation"] = poly
                    break  # Use first valid polygon

        # Handle keypoints
        if det.keypoints:
            label_kps = labels[class_idx].get("keypoints", [])
            kps = []
            for kp in det.keypoints:
                if not kp.visibility:
                    continue
                kp_id = next((k["id"] for k in label_kps if k["name"] == kp.name), None)
                if kp_id is not None:
                    kps.append({"id": kp_id, "point": [float(kp.x) / img_w, float(kp.y) / img_h]})
            if kps:
                obj["keypoints"] = kps

        objects.append(obj)

    return objects


@predict_api.route("/api/predict/run", methods=["POST"])
def predict_run():
    """Inference on an uploaded image — no project/file_id needed."""
    training_id = request.form.get("training_id")
    threshold = float(request.form.get("threshold", 0.5))

    if not training_id:
        return jsonify({"error": "training_id is required"}), 400
    if "file" not in request.files:
        return jsonify({"error": "file is required"}), 400

    training = get_training(training_id)
    if not training:
        return jsonify({"error": "Training not found"}), 404
    if training["status"] != "done":
        return jsonify({"error": f"Training is not done (status: {training['status']})"}), 400
    if not training["model_path"]:
        return jsonify({"error": "No model checkpoint available"}), 400

    training["config"] = json.loads(training["config"])
    labels = training["config"].get("labels", [])
    if not labels:
        return jsonify({"error": "Training config has no labels"}), 400

    try:
        image = Image.open(request.files["file"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {e}"}), 400

    img_w, img_h = image.size
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    try:
        model = load_model_from_training(training)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    try:
        detections = model.predict(image_np)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    # Draw detections on image using pixelflow annotators
    annotated = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    if any(det.masks for det in detections):
        annotated = pf.annotators.mask(annotated, detections)
    annotated = pf.annotators.box(annotated, detections)
    annotated = pf.annotators.label(annotated, detections)
    if any(det.keypoints for det in detections):
        annotated = pf.annotators.keypoint(annotated, detections)

    buf = BytesIO()
    Image.fromarray(annotated).save(buf, format="JPEG", quality=90)
    annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    result = []
    for det in detections:
        entry = {
            "class_name": det.class_name,
            "bbox": [float(x) for x in det.bbox],
            "confidence": round(float(det.confidence), 4) if det.confidence is not None else None,
        }
        if det.keypoints:
            entry["keypoints"] = [
                {"name": kp.name, "x": kp.x, "y": kp.y, "visible": kp.visibility}
                for kp in det.keypoints
            ]
        result.append(entry)

    return jsonify({
        "detections": result,
        "annotated_image": annotated_b64,
        "image_width": img_w,
        "image_height": img_h,
    })


@predict_api.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    training_id = body.get("training_id")
    file_id = body.get("file_id")
    threshold = float(body.get("threshold", 0.5))

    if not training_id or not file_id:
        return jsonify({"error": "training_id and file_id are required"}), 400

    training = get_training(training_id)
    if not training:
        return jsonify({"error": "Training not found"}), 404
    if training["status"] != "done":
        return jsonify({"error": f"Training is not done (status: {training['status']})"}), 400
    if not training["model_path"]:
        return jsonify({"error": "No model checkpoint available for this training"}), 400

    file_row = get_file_by_id(file_id)
    if not file_row:
        return jsonify({"error": "File not found"}), 404

    training["config"] = json.loads(training["config"])
    labels = training["config"].get("labels", [])
    if not labels:
        return jsonify({"error": "Training config has no labels"}), 400

    try:
        model = load_model_from_training(training)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    img_path = file_path(file_row["filename"])
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {e}"}), 500

    img_w, img_h = image.size
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    try:
        detections = model.predict(image_np)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    objects = detections_to_objects(detections, labels, img_w, img_h)
    return jsonify({"objects": objects})
