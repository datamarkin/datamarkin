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

from config import file_path
from queries import get_file_by_id, get_training
from mozo import ModelManager

predict_api = Blueprint("predict_api", __name__)

# Singleton model manager for lazy loading and memory management
model_manager = ModelManager()


def _load_model(architecture, training_id, training, cfg, project_type, class_names):
    """Load a model via mozo, dispatching to the right family based on architecture."""
    if architecture == "detectron2":
        return model_manager.get_model(
            'detectron2',
            training_id,
            checkpoint_path=training["model_path"],
            variant=cfg.get("variant", "mask_rcnn_R_50_FPN_3x"),
            class_names=class_names,
            num_classes=len(class_names),
            device='cpu',
        )
    # Default: RF-DETR
    return model_manager.get_model(
        'rfdetr',
        training_id,
        checkpoint_path=training["model_path"],
        model_size=cfg.get("model_size", "base"),
        project_type=project_type,
        resolution=cfg.get("resolution", 560),
        class_names=class_names,
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

    cfg = json.loads(training["config"])
    labels = cfg.get("labels", [])
    if not labels:
        return jsonify({"error": "Training config has no labels"}), 400

    try:
        image = Image.open(request.files["file"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {e}"}), 400

    img_w, img_h = image.size

    project_type = cfg.get("project_type", "detection")
    project_type = {"object_detection": "detection", "instance_segmentation": "segmentation"}.get(project_type, project_type)

    class_names = [label["name"] for label in labels]
    model_architecture = cfg.get("model_architecture", "rfdetr")

    try:
        model = _load_model(model_architecture, training_id, training, cfg, project_type, class_names)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    try:
        detections = model.predict(image, threshold=threshold)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    # Draw detections on image using pixelflow annotators
    annotated = np.array(image)
    if any(det.masks for det in detections):
        annotated = pf.annotators.mask(annotated, detections)
    annotated = pf.annotators.box(annotated, detections)
    annotated = pf.annotators.label(annotated, detections)

    buf = BytesIO()
    Image.fromarray(annotated).save(buf, format="JPEG", quality=90)
    annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    result = []
    for det in detections:
        result.append({
            "class_name": det.class_name,
            "bbox": [float(x) for x in det.bbox],
            "confidence": round(float(det.confidence), 4) if det.confidence is not None else None,
        })

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

    cfg = json.loads(training["config"])
    labels = cfg.get("labels", [])
    if not labels:
        return jsonify({"error": "Training config has no labels"}), 400

    class_names = [label["name"] for label in labels]

    project_type = cfg.get("project_type", "detection")
    project_type = {"object_detection": "detection", "instance_segmentation": "segmentation"}.get(project_type, project_type)
    model_architecture = cfg.get("model_architecture", "rfdetr")

    try:
        model = _load_model(model_architecture, training_id, training, cfg, project_type, class_names)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    img_path = file_path(file_row["filename"])
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {e}"}), 500

    img_w, img_h = image.size

    try:
        detections = model.predict(image, threshold=threshold)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    objects = detections_to_objects(detections, labels, img_w, img_h)
    return jsonify({"objects": objects})
