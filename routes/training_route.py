import io
import json
import os
import signal
import subprocess
import sys
import zipfile
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from config import APP_DIR, TRAINING_JOBS_DIR, FILES_DIR
from db import get_db, new_id, now
from queries import (
    get_project_by_id,
    get_project_trainings,
    get_training,
    update_training_status,
)

training_api = Blueprint("training_api", __name__)


# ── COCO dataset preparation ──────────────────────────────────────────────────

def _prepare_coco_dataset(training_id: str, project_id: str, db) -> str:
    """
    Build COCO-format dataset from the project's annotated files.
    Creates symlinks to original images; writes _annotations.coco.json per split.
    Returns the dataset_dir path string.

    Annotation format in DB (normalized, 0-1):
      {"objects": [{"class": 0, "bbox": [x_topleft, y_topleft, w, h]}]}
    COCO bbox format: [x_min, y_min, w, h] in absolute pixels.
    """
    project = get_project_by_id(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    labels = json.loads(project["labels"] or "[]")
    if not labels:
        raise ValueError("Project has no labels defined")

    # COCO categories are 1-indexed; our label IDs start at 0
    categories = [{"id": lbl["id"] + 1, "name": lbl["name"]} for lbl in labels]

    dataset_dir = TRAINING_JOBS_DIR / training_id / "dataset"

    rows = db.execute(
        "SELECT id, filename, extension, width, height, split, annotations "
        "FROM files WHERE project_id = ? "
        "AND annotations IS NOT NULL AND annotations != '' AND annotations != 'null'",
        (project_id,),
    ).fetchall()

    if not rows:
        raise ValueError("No annotated files found in this project")

    splits = {"train": [], "valid": []}
    for row in rows:
        s = row["split"] if row["split"] in splits else "train"
        splits[s].append(row)

    if not splits["valid"] and splits["train"]:
        n_valid = max(1, len(splits["train"]) // 5)  # 20%
        splits["valid"] = splits["train"][-n_valid:]
        splits["train"] = splits["train"][:-n_valid]

    for split_name, files in splits.items():
        if not files:
            continue
        split_dir = dataset_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        coco_images = []
        coco_annotations = []
        ann_id = 1

        for img_idx, f in enumerate(files, start=1):
            img_filename = f["filename"]
            src_path = FILES_DIR / img_filename[:3] / img_filename
            dst_path = split_dir / img_filename

            if not dst_path.exists():
                os.symlink(src_path, dst_path)

            coco_images.append({
                "id": img_idx,
                "file_name": img_filename,
                "width": f["width"],
                "height": f["height"],
            })

            ann_data = json.loads(f["annotations"] or "{}")
            for obj in ann_data.get("objects", []):
                bbox_norm = obj.get("bbox")
                if not bbox_norm or len(bbox_norm) != 4:
                    continue
                x, y, w, h = bbox_norm
                x_abs = x * f["width"]
                y_abs = y * f["height"]
                w_abs = w * f["width"]
                h_abs = h * f["height"]

                coco_ann = {
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": obj["class"] + 1,
                    "bbox": [x_abs, y_abs, w_abs, h_abs],
                    "area": w_abs * h_abs,
                    "iscrowd": 0,
                }
                seg_norm = obj.get("segmentation")
                if seg_norm and len(seg_norm) >= 6:
                    abs_seg = []
                    for i in range(0, len(seg_norm) - 1, 2):
                        abs_seg.append(seg_norm[i] * f["width"])
                        abs_seg.append(seg_norm[i + 1] * f["height"])
                    coco_ann["segmentation"] = [abs_seg]
                else:
                    coco_ann["segmentation"] = []
                coco_annotations.append(coco_ann)
                ann_id += 1

        coco_json = {
            "info": {"description": f"Datamarkin export — {project['name']}"},
            "categories": categories,
            "images": coco_images,
            "annotations": coco_annotations,
        }

        with open(split_dir / "_annotations.coco.json", "w") as fh:
            json.dump(coco_json, fh)

    return str(dataset_dir)


# ── Task dispatcher ───────────────────────────────────────────────────────────

def _launch_worker(training_id: str, db) -> None:
    log_path = TRAINING_JOBS_DIR / training_id / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, str(APP_DIR / "scripts" / "training_worker.py"),
         "--training-id", training_id],
        cwd=str(APP_DIR),
        stdout=log_file,
        stderr=log_file,
    )
    db.execute(
        "UPDATE trainings SET status='running', pid=?, updated_at=? WHERE id=?",
        (proc.pid, now(), training_id),
    )
    db.commit()


def _maybe_dispatch(db) -> None:
    running = db.execute(
        "SELECT id, pid FROM trainings WHERE status='running'"
    ).fetchone()
    if running:
        pid = running["pid"]
        alive = False
        if pid:
            try:
                os.kill(pid, 0)  # signal 0 = existence check, no actual signal
                alive = True
            except (ProcessLookupError, PermissionError):
                alive = False
        if alive:
            return
        # Process is dead but status is still 'running' — clean up
        db.execute(
            "UPDATE trainings SET status='failed', error='Worker process died unexpectedly', updated_at=? WHERE id=?",
            (now(), running["id"]),
        )
        db.commit()

    nxt = db.execute(
        "SELECT id FROM trainings WHERE status='pending' ORDER BY created_at LIMIT 1"
    ).fetchone()
    if nxt:
        _launch_worker(nxt["id"], db)


# ── Routes ────────────────────────────────────────────────────────────────────

@training_api.route("/api/training/start", methods=["POST"])
def training_start():
    body = request.get_json(force=True)
    project_id = body.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id required"}), 400

    project = get_project_by_id(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    training_id = new_id()

    cfg = json.loads(project["configuration"] or "{}")

    config_snapshot = {
        "model_size":              cfg.get("model_size", "base"),
        "epochs":                  cfg.get("epochs", 20),
        "batch_size":              cfg.get("batch_size", 4),
        "resolution":              cfg.get("resolution", 560),
        "lr":                      cfg.get("lr", 1e-4),
        "early_stopping":          cfg.get("early_stopping", True),
        "early_stopping_patience": cfg.get("early_stopping_patience", 3),
        "augmentation":  json.loads(project["augmentation"]  or "[]"),
        "preprocessing": json.loads(project["preprocessing"] or "[]"),
        "labels":        json.loads(project["labels"]        or "[]"),
        "project_type":  project["type"],
    }

    db = get_db()
    try:
        dataset_dir = _prepare_coco_dataset(training_id, project_id, db)
    except ValueError as exc:
        db.close()
        return jsonify({"error": str(exc)}), 422

    config_snapshot["dataset_dir"] = dataset_dir

    db.execute(
        """INSERT INTO trainings (id, project_id, status, config, created_at, updated_at)
           VALUES (?, ?, 'pending', ?, ?, ?)""",
        (training_id, project_id, json.dumps(config_snapshot), now(), now()),
    )
    db.commit()

    _maybe_dispatch(db)
    db.close()

    return jsonify({"training_id": training_id}), 201


@training_api.route("/api/training/<training_id>", methods=["GET"])
def training_status(training_id):
    training = get_training(training_id)
    if not training:
        return jsonify({"error": "Not found"}), 404

    if training["status"] in ("done", "failed", "stopped"):
        db = get_db()
        _maybe_dispatch(db)
        db.close()

    return jsonify(training)


@training_api.route("/api/training/<training_id>/live", methods=["GET"])
def training_live(training_id):
    path = TRAINING_JOBS_DIR / training_id / "live.json"
    if not path.exists():
        return jsonify({}), 200
    try:
        return jsonify(json.loads(path.read_text()))
    except Exception:
        return jsonify({}), 200


@training_api.route("/api/training/<training_id>/log", methods=["GET"])
def training_log(training_id):
    path = TRAINING_JOBS_DIR / training_id / "worker.log"
    if not path.exists():
        return "", 200
    try:
        return path.read_text(), 200, {"Content-Type": "text/plain"}
    except Exception:
        return "", 200


@training_api.route("/api/training/<training_id>/stop", methods=["POST"])
def training_stop(training_id):
    training = get_training(training_id)
    if not training:
        return jsonify({"error": "Not found"}), 404

    if training["status"] not in ("running", "pending"):
        return jsonify({"error": "Not running"}), 409

    pid = training.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    update_training_status(training_id, "stopped")

    db = get_db()
    _maybe_dispatch(db)
    db.close()

    return jsonify({"status": "stopped"})


@training_api.route("/api/projects/<project_id>/trainings", methods=["GET"])
def project_trainings(project_id):
    if not get_project_by_id(project_id):
        return jsonify({"error": "Project not found"}), 404
    return jsonify(get_project_trainings(project_id))


@training_api.route("/api/projects/<project_id>/export/coco", methods=["GET"])
def export_coco(project_id):
    project = get_project_by_id(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    labels = json.loads(project["labels"] or "[]")
    categories = [{"id": lbl["id"] + 1, "name": lbl["name"]} for lbl in labels]

    db = get_db()
    rows = db.execute(
        "SELECT filename, width, height, split, annotations "
        "FROM files WHERE project_id = ? "
        "AND annotations IS NOT NULL AND annotations != '' AND annotations != 'null'",
        (project_id,),
    ).fetchall()
    db.close()

    if not rows:
        return jsonify({"error": "No annotated files found"}), 422

    splits = {"train": [], "valid": [], "test": []}
    for row in rows:
        split = row["split"] if row["split"] in splits else "train"
        splits[split].append(row)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for split_name, files in splits.items():
            if not files:
                continue

            coco_images = []
            coco_annotations = []
            ann_id = 1

            for img_idx, f in enumerate(files, start=1):
                img_filename = f["filename"]
                src_path = FILES_DIR / img_filename[:3] / img_filename

                if src_path.exists():
                    zf.write(src_path, f"{split_name}/{img_filename}")

                coco_images.append({
                    "id": img_idx,
                    "file_name": img_filename,
                    "width": f["width"],
                    "height": f["height"],
                })

                ann_data = json.loads(f["annotations"] or "{}")
                for obj in ann_data.get("objects", []):
                    bbox_norm = obj.get("bbox")
                    if not bbox_norm or len(bbox_norm) != 4:
                        continue
                    x, y, w, h = bbox_norm
                    x_abs = x * f["width"]
                    y_abs = y * f["height"]
                    w_abs = w * f["width"]
                    h_abs = h * f["height"]
                    coco_ann = {
                        "id": ann_id,
                        "image_id": img_idx,
                        "category_id": obj["class"] + 1,
                        "bbox": [x_abs, y_abs, w_abs, h_abs],
                        "area": w_abs * h_abs,
                        "iscrowd": 0,
                    }
                    seg_norm = obj.get("segmentation")
                    if seg_norm and len(seg_norm) >= 6:
                        abs_seg = []
                        for i in range(0, len(seg_norm) - 1, 2):
                            abs_seg.append(seg_norm[i] * f["width"])
                            abs_seg.append(seg_norm[i + 1] * f["height"])
                        coco_ann["segmentation"] = [abs_seg]
                    else:
                        coco_ann["segmentation"] = []
                    coco_annotations.append(coco_ann)
                    ann_id += 1

            zf.writestr(
                f"{split_name}/_annotations.coco.json",
                json.dumps({
                    "info": {"description": project["name"]},
                    "categories": categories,
                    "images": coco_images,
                    "annotations": coco_annotations,
                }, indent=2),
            )

    buf.seek(0)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project["name"])
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{safe_name}_coco.zip",
    )
