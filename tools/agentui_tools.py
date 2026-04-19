"""AgentUI tool definitions for Datamarkin.

Registered at app startup via agentui.register_tool().
"""

import json
import os
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

from agentui.core.tool import InputTool, Port, PortType, Tool, ToolOutput

try:
    from routes.predict_route import load_model_from_training
    PREDICT_AVAILABLE = True
except ImportError:
    PREDICT_AVAILABLE = False

# Colour palette for auto-created labels
_LABEL_COLORS = [
    "e6194b", "3cb44b", "ffe119", "4363d8", "f58231",
    "911eb4", "42d4f4", "f032e6", "bfef45", "fabed4",
    "469990", "dcbeff", "9a6324", "fffac8", "800000",
    "aaffc3", "808000", "ffd8b1", "000075", "a9a9a9",
]


def _annotations_to_detections(annotations, labels, img_w, img_h):
    """Convert Datamarkin annotation dict to pixelflow Detections.

    Reverse of ``detections_to_objects`` in predict_route.
    """
    from pixelflow.detections import Detection, Detections, KeyPoint

    result = Detections()
    if not annotations or "objects" not in annotations:
        return result

    label_by_id = {l["id"]: l for l in labels}

    for obj in annotations["objects"]:
        label_id = obj.get("class")
        label = label_by_id.get(label_id)
        if label is None:
            continue

        # bbox stored as [x, y, w, h] normalised
        bx, by, bw, bh = obj["bbox"]
        bbox = [
            bx * img_w,
            by * img_h,
            (bx + bw) * img_w,
            (by + bh) * img_h,
        ]

        segments = None
        if "segmentation" in obj:
            seg = obj["segmentation"]
            # Denormalise flat polygon to absolute pixel coords, keep as polygon
            segments = [[seg[i] * img_w, seg[i + 1] * img_h] for i in range(0, len(seg), 2)]

        keypoints = None
        if "keypoints" in obj:
            label_kps = {k["id"]: k for k in label.get("keypoints", [])}
            keypoints = []
            for kp in obj["keypoints"]:
                kp_def = label_kps.get(kp["id"])
                name = kp_def["name"] if kp_def else str(kp["id"])
                px, py = kp["point"]
                keypoints.append(KeyPoint(px * img_w, py * img_h, name, True))

        det = Detection(
            bbox=bbox,
            class_id=label_id,
            class_name=label["name"],
            confidence=obj.get("confidence"),
            segments=segments,
            keypoints=keypoints or None,
        )
        result.add_detection(det)

    return result


def _detections_to_annotation_objects(detections, target_labels, img_w, img_h):
    """Convert pixelflow Detections to annotation objects using class_name mapping.

    Unlike ``detections_to_objects`` (which uses positional class_id), this
    matches by ``class_name`` against *target_labels* so it works across
    projects with different label sets.

    Returns (objects_list, updated_labels) — labels may have new entries if
    auto-creation was triggered.
    """
    from routes.predict_route import mask_to_norm_polygon

    if detections is None or len(detections) == 0:
        return [], target_labels

    labels = list(target_labels)  # don't mutate caller's list
    name_to_label = {l["name"].lower(): l for l in labels}

    objects = []
    for det in detections:
        class_name = (det.class_name or "").strip()
        if not class_name:
            continue

        label = name_to_label.get(class_name.lower())
        if label is None:
            # Auto-create label
            next_id = max((l["id"] for l in labels), default=-1) + 1
            color = _LABEL_COLORS[next_id % len(_LABEL_COLORS)]
            label = {"id": next_id, "name": class_name, "color": color}
            labels.append(label)
            name_to_label[class_name.lower()] = label

        x1, y1, x2, y2 = det.bbox
        norm_bbox = [
            float(x1) / img_w,
            float(y1) / img_h,
            float(x2 - x1) / img_w,
            float(y2 - y1) / img_h,
        ]

        obj = {"class": label["id"], "bbox": norm_bbox}
        if det.confidence is not None:
            obj["confidence"] = round(float(det.confidence), 4)

        if det.segments:
            # Lossless: polygon coords stored directly (from DatasetInput)
            poly = []
            for pt in det.segments:
                poly.append(float(pt[0]) / img_w)
                poly.append(float(pt[1]) / img_h)
            if len(poly) >= 6:
                obj["segmentation"] = poly
        elif det.masks:
            # Lossy: extract polygon from boolean mask (from model inference)
            for mask in det.masks:
                poly = mask_to_norm_polygon(mask, img_w, img_h)
                if poly:
                    obj["segmentation"] = poly
                    break

        if det.keypoints:
            label_kps = label.get("keypoints", [])
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

    return objects, labels


class DatamarkinLocalModel(Tool):
    """Run inference using a locally trained Datamarkin model."""

    @property
    def tool_type(self) -> str:
        return "DatamarkinLocalModel"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Input image")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"detections": Port("detections", PortType.DETECTIONS, "Detected objects")}

    def get_parameter_options(self) -> dict:
        """Populate the training_id dropdown from the live database."""
        from queries import get_done_trainings, get_project_by_id
        trainings = get_done_trainings()
        options = []
        for t in trainings:
            config = json.loads(t.get("config") or "{}")
            project = get_project_by_id(t["project_id"]) if t.get("project_id") else None
            project_name = project["name"] if project else config.get("name", "Built-in Model")
            arch = config.get("model_architecture", "rfdetr")
            if arch == "detectron2":
                label = f"{project_name} ({config.get('variant', 'detectron2')})"
            else:
                model_size = config.get("model_size", "base")
                metrics = json.loads(t.get("metrics") or "{}")
                map_val = metrics.get("mAP50", "")
                label = f"{project_name} ({model_size})"
                if map_val:
                    label += f" — mAP50: {map_val:.2f}"
            options.append({"value": t["id"], "label": label})
        return {
            "training_id": {
                "type": "select",
                "options": options,
            }
        }

    def process(self) -> bool:
        if not PREDICT_AVAILABLE:
            print("DatamarkinLocalModel: predict route not available")
            return False
        if "image" not in self.inputs:
            print("DatamarkinLocalModel: no input image")
            return False

        training_id = self.parameters.get("training_id", "").strip()
        if not training_id:
            print("DatamarkinLocalModel: no training_id selected")
            return False

        from queries import get_training
        training = get_training(training_id)
        if not training or training["status"] != "done":
            print(f"DatamarkinLocalModel: training {training_id!r} not found or not complete")
            return False

        training["config"] = json.loads(training.get("config") or "{}")

        try:
            pil_image = self.inputs["image"].data
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            model = load_model_from_training(training)
            detections = model.predict(cv2_image)
            self.outputs["detections"] = ToolOutput(detections, PortType.DETECTIONS)
            return True

        except Exception as e:
            import traceback
            print(f"DatamarkinLocalModel error: {e}")
            traceback.print_exc()
            return False


METADATA = {
    "name": "My Trained Model",
    "category": "My Models",
    "description": "Run inference using your locally trained Datamarkin models.",
    "parameters": {
        "training_id": "",
    },
}


# ── DatasetInput ─────────────────────────────────────────────────────────────


class DatasetInput(InputTool):
    """Load images and annotations from a Datamarkin project."""

    @property
    def tool_type(self) -> str:
        return "DatasetInput"

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {
            "images": Port("images", PortType.IMAGE, "List of PIL images"),
            "detections": Port("detections", PortType.DETECTIONS, "List of pixelflow Detections"),
        }

    def get_parameter_options(self) -> dict:
        from queries import get_all_projects
        projects = get_all_projects()
        options = [{"value": p["id"], "label": p["name"]} for p in projects]
        return {
            "project_id": {"type": "select", "options": options},
            "split": {
                "type": "select",
                "options": [
                    {"value": "all", "label": "All"},
                    {"value": "train", "label": "Train"},
                    {"value": "valid", "label": "Valid"},
                    {"value": "test", "label": "Test"},
                ],
            },
            "has_annotations": {
                "type": "select",
                "options": [
                    {"value": "any", "label": "Any"},
                    {"value": "yes", "label": "Annotated only"},
                    {"value": "no", "label": "Unannotated only"},
                ],
            },
            "limit": {"type": "number", "min": 0, "step": 1},
        }

    def process(self) -> bool:
        from config import file_path as get_file_path
        from db_models import File, Project
        from queries import get_project_by_id, get_project_files

        project_id = self.parameters.get("project_id", "").strip()
        if not project_id:
            print("DatasetInput: no project selected")
            return False

        project = get_project_by_id(project_id)
        if not project:
            print(f"DatasetInput: project {project_id!r} not found")
            return False
        project = Project(project)
        labels = project.labels or []

        files = get_project_files(project_id)
        files = [File(f) for f in files]

        # Apply filters
        split_filter = self.parameters.get("split", "all")
        if split_filter != "all":
            files = [f for f in files if f.get("split") == split_filter]

        ann_filter = self.parameters.get("has_annotations", "any")
        if ann_filter == "yes":
            files = [f for f in files if f.annotations and f.annotations.get("objects")]
        elif ann_filter == "no":
            files = [f for f in files if not f.annotations or not f.annotations.get("objects")]

        limit = int(self.parameters.get("limit", 0) or 0)
        if limit > 0:
            files = files[:limit]

        if not files:
            print("DatasetInput: no files match the filters")
            return False

        images = []
        detections_list = []

        for f in files:
            path = get_file_path(f["filename"])
            if not os.path.exists(path):
                print(f"DatasetInput: file not found on disk: {path}")
                continue

            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"DatasetInput: cannot open {path}: {e}")
                continue

            img_w, img_h = img.size
            dets = _annotations_to_detections(f.annotations, labels, img_w, img_h)
            # Embed file_id so DatasetSave can use it in update mode
            for det in dets:
                if det.metadata is None:
                    det.metadata = {}
                det.metadata["file_id"] = f["id"]

            images.append(img)
            detections_list.append(dets)

        if not images:
            print("DatasetInput: could not load any images")
            return False

        self.outputs["images"] = ToolOutput(images, PortType.IMAGE)
        self.outputs["detections"] = ToolOutput(detections_list, PortType.DETECTIONS)
        return True


DATASET_INPUT_METADATA = {
    "name": "Dataset Input",
    "category": "Datamarkin",
    "description": "Load images and annotations from a Datamarkin project.",
    "parameters": {
        "project_id": "",
        "split": "all",
        "has_annotations": "any",
        "limit": 0,
    },
}


# ── DatasetSave ──────────────────────────────────────────────────────────────


class DatasetSave(Tool):
    """Save images and annotations to a Datamarkin project."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._resolved_project_id: Optional[str] = None
        self._target_labels: Optional[list] = None
        self._labels_before_count: int = 0

    @property
    def tool_type(self) -> str:
        return "DatasetSave"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {
            "image": Port("image", PortType.IMAGE, "Image to save"),
            "detections": Port("detections", PortType.DETECTIONS, "Detections for this image"),
        }

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {}

    def get_parameter_options(self) -> dict:
        from queries import get_all_projects
        projects = get_all_projects()
        options = [{"value": "__new__", "label": "+ Create New Project"}]
        options += [{"value": p["id"], "label": p["name"]} for p in projects]
        return {
            "target_project_id": {"type": "select", "options": options},
            "new_project_type": {
                "type": "select",
                "options": [
                    {"value": "object_detection", "label": "Object Detection"},
                    {"value": "segmentation", "label": "Segmentation"},
                    {"value": "keypoint-detection", "label": "Keypoint Detection"},
                ],
            },
            "save_mode": {
                "type": "select",
                "options": [
                    {"value": "create_files", "label": "Create new files"},
                    {"value": "update_annotations", "label": "Update annotations on existing files"},
                ],
            },
        }

    def _resolve_project(self) -> str | None:
        """Get or create the target project. Called once per batch."""
        if self._resolved_project_id:
            return self._resolved_project_id

        from queries import create_project, get_project_by_id

        target = self.parameters.get("target_project_id", "").strip()

        if target == "__new__":
            name = self.parameters.get("new_project_name", "").strip()
            if not name:
                name = "Workflow Output"
            ptype = self.parameters.get("new_project_type", "object_detection")
            pid = create_project(name, ptype)
            self._resolved_project_id = pid
            self._target_labels = []
            self._labels_before_count = 0
            return pid

        if not target:
            print("DatasetSave: no target project selected")
            return None

        project = get_project_by_id(target)
        if not project:
            print(f"DatasetSave: project {target!r} not found")
            return None

        from db_models import Project
        project = Project(project)
        self._resolved_project_id = target
        self._target_labels = list(project.labels or [])
        self._labels_before_count = len(self._target_labels)
        return target

    def _flush_labels_if_changed(self):
        """Persist auto-created labels to the DB when new ones were added."""
        if self._target_labels and len(self._target_labels) > self._labels_before_count:
            from queries import update_project_labels
            update_project_labels(self._resolved_project_id, self._target_labels)
            self._labels_before_count = len(self._target_labels)

    def process(self) -> bool:
        from config import file_path as get_file_path
        from db import new_id
        from queries import (
            get_file_by_id,
            insert_file,
            update_file_annotations,
        )

        project_id = self._resolve_project()
        if not project_id:
            return False

        save_mode = self.parameters.get("save_mode", "create_files")
        detections = self.inputs["detections"].data if "detections" in self.inputs else None

        if save_mode == "create_files":
            if "image" not in self.inputs:
                print("DatasetSave: no input image")
                return False

            pil_image = self.inputs["image"].data
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            img_w, img_h = pil_image.size
            file_id = new_id()
            ext = ".jpg"
            filename = f"{file_id}{ext}"
            dest = get_file_path(filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pil_image.save(str(dest), quality=95)
            filesize = os.path.getsize(dest)

            insert_file(file_id, project_id, filename, ext, img_w, img_h, filesize)

            if detections and len(detections) > 0:
                objects, self._target_labels = _detections_to_annotation_objects(
                    detections, self._target_labels, img_w, img_h,
                )
                if objects:
                    update_file_annotations(file_id, json.dumps({"objects": objects}))
                    self._flush_labels_if_changed()

            return True

        elif save_mode == "update_annotations":
            # Extract file_id from detection metadata (set by DatasetInput)
            file_id = None
            if detections and len(detections) > 0:
                first_det = detections[0]
                if first_det.metadata:
                    file_id = first_det.metadata.get("file_id")

            if not file_id:
                print("DatasetSave: update mode requires detections with file_id in metadata (from DatasetInput)")
                return False

            file_row = get_file_by_id(file_id)
            if not file_row:
                print(f"DatasetSave: file {file_id!r} not found")
                return False

            img_w = file_row["width"]
            img_h = file_row["height"]

            objects, self._target_labels = _detections_to_annotation_objects(
                detections, self._target_labels, img_w, img_h,
            )
            update_file_annotations(file_id, json.dumps({"objects": objects}))
            self._flush_labels_if_changed()

            return True

        print(f"DatasetSave: unknown save_mode {save_mode!r}")
        return False


DATASET_SAVE_METADATA = {
    "name": "Dataset Save",
    "category": "Datamarkin",
    "description": "Save images and annotations to a Datamarkin project. Can create new projects or update existing ones.",
    "parameters": {
        "target_project_id": "",
        "new_project_name": "",
        "new_project_type": "object_detection",
        "save_mode": "create_files",
    },
}
