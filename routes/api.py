import json

from flask import Blueprint, jsonify, request

from queries import (
    get_all_projects,
    get_project_by_id,
    create_project,
    get_project_files_paginated,
    get_file_by_id,
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
