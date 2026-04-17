import json
import shutil

from flask import Blueprint, jsonify, request

from config import FILES_DIR, MODELS_DIR, TRAINING_JOBS_DIR
from thumbnails import THUMBS_DIR, PRESETS
from queries import (
    get_all_projects,
    get_project_by_id,
    create_project,
    get_project_files_paginated,
    get_file_by_id,
    update_file_annotations,
    clear_project_annotations,
    get_project_file_paths,
    delete_project_files,
    get_project_training_ids,
    delete_project,
    has_active_training,
)

api = Blueprint("api", __name__, url_prefix="/api")


def api_response(data, status=200):
    return jsonify({"data": data}), status


def api_error(message, code, status):
    return jsonify({"error": {"message": message, "code": code}}), status


@api.route("/projects", methods=["GET"])
def list_projects():
    projects = get_all_projects()
    return api_response(projects)


@api.route("/projects", methods=["POST"])
def create_project_endpoint():
    body = request.get_json(silent=True)
    if not body:
        return api_error("Request body must be JSON", "invalid_body", 400)

    name = body.get("name")
    project_type = body.get("type")
    if not name or not project_type:
        return api_error("'name' and 'type' are required", "missing_fields", 400)

    labels = body.get("labels")
    labels_json = json.dumps(labels) if labels is not None else None

    project_id = create_project(name, project_type, labels_json)
    project = get_project_by_id(project_id)
    return api_response(project, 201)


@api.route("/projects/<project_id>", methods=["GET"])
def get_project(project_id):
    project = get_project_by_id(project_id)
    if not project:
        return api_error("Project not found", "not_found", 404)
    return api_response(project)


@api.route("/files", methods=["GET"])
def list_files():
    project_id = request.args.get("project_id")
    if not project_id:
        return api_error("'project_id' query parameter is required", "missing_filter", 400)

    project = get_project_by_id(project_id)
    if not project:
        return api_error("Project not found", "not_found", 404)

    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 100, type=int)
    split = request.args.get("split")
    has_annotations_raw = request.args.get("has_annotations")

    has_annotations = None
    if has_annotations_raw is not None:
        has_annotations = has_annotations_raw.lower() in ("true", "1", "yes")

    result = get_project_files_paginated(
        project_id,
        page=page,
        per_page=per_page,
        split=split,
        has_annotations=has_annotations,
    )

    return jsonify({
        "data": result["items"],
        "pagination": {
            "total": result["total"],
            "page": result["page"],
            "per_page": result["per_page"],
            "total_pages": result["total_pages"],
            "has_next": result["has_next"],
            "has_prev": result["has_prev"],
        },
    })


@api.route("/files/<file_id>", methods=["GET"])
def get_file(file_id):
    file = get_file_by_id(file_id)
    if not file:
        return api_error("File not found", "not_found", 404)
    return api_response(file)


@api.route("/files/<file_id>", methods=["PATCH"])
def patch_file(file_id):
    file = get_file_by_id(file_id)
    if not file:
        return api_error("File not found", "not_found", 404)

    body = request.get_json(silent=True) or {}
    if "annotations" not in body:
        return api_error("'annotations' is required", "missing_fields", 400)

    annotations_json = json.dumps(body["annotations"]) if body["annotations"] is not None else None
    update_file_annotations(file_id, annotations_json)
    return api_response({"saved": True})


# ── Danger zone ──────────────────────────────────────────────────────────────

def _cleanup_file_assets(file_id, filename):
    prefix = filename[:3]
    (FILES_DIR / prefix / filename).unlink(missing_ok=True)
    try:
        (FILES_DIR / prefix).rmdir()
    except OSError:
        pass
    for preset, cfg in PRESETS.items():
        if cfg["save"]:
            (THUMBS_DIR / preset / f"{file_id}.jpg").unlink(missing_ok=True)


def _cleanup_training_assets(training_id):
    (MODELS_DIR / f"{training_id}.pth").unlink(missing_ok=True)
    job_dir = TRAINING_JOBS_DIR / training_id
    if job_dir.is_dir():
        shutil.rmtree(job_dir, ignore_errors=True)


@api.route("/projects/<project_id>/annotations", methods=["DELETE"])
def delete_all_annotations(project_id):
    project = get_project_by_id(project_id)
    if not project:
        return api_error("Project not found", "not_found", 404)
    count = clear_project_annotations(project_id)
    return api_response({"deleted_annotations_from": count})


@api.route("/projects/<project_id>/files", methods=["DELETE"])
def delete_all_files(project_id):
    project = get_project_by_id(project_id)
    if not project:
        return api_error("Project not found", "not_found", 404)
    if has_active_training(project_id):
        return api_error(
            "Cannot delete files while a training is running. Stop it first.",
            "training_active", 409,
        )
    file_records = get_project_file_paths(project_id)
    count = delete_project_files(project_id)
    for file_id, filename in file_records:
        _cleanup_file_assets(file_id, filename)
    return api_response({"deleted_files": count})


@api.route("/projects/<project_id>", methods=["DELETE"])
def delete_project_endpoint(project_id):
    project = get_project_by_id(project_id)
    if not project:
        return api_error("Project not found", "not_found", 404)
    body = request.get_json(silent=True) or {}
    if (body.get("name") or "").strip() != project["name"]:
        return api_error(
            "Project name does not match. Type the exact project name to confirm.",
            "name_mismatch", 400,
        )
    if has_active_training(project_id):
        return api_error(
            "Cannot delete project while a training is running. Stop it first.",
            "training_active", 409,
        )
    file_records = get_project_file_paths(project_id)
    training_ids = get_project_training_ids(project_id)
    delete_project(project_id)
    for file_id, filename in file_records:
        _cleanup_file_assets(file_id, filename)
    for tid in training_ids:
        _cleanup_training_assets(tid)
    return api_response({"deleted": True})
