import json
import os

from flask import render_template, abort, request, redirect, url_for, jsonify
from PIL import Image

from config import file_path as get_file_path
from db import new_id
from queries import get_all_projects, get_project_by_id, get_project_files, get_project_files_paginated, get_file_by_id, create_project, insert_file


def projects_page_route(app_name):
    return render_template(
        "projects.html",
        app_name="Datamarkin",
        active_tab="projects",
        projects=get_all_projects(),
    )


def project_new_page_route():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        project_type = request.form.get("type", "object_detection")
        labels = request.form.get("labels") or None
        project_id = create_project(name, project_type, labels)
        return redirect(url_for("project", project_id=project_id))

    return render_template(
        "project_new.html",
        app_name="Datamarkin",
        active_tab="projects",
    )


def project_page_route(project_id: str):
    project = get_project_by_id(project_id)
    if not project:
        abort(404)

    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 100, type=int)
    split = request.args.get("split", None)

    has_annotations_raw = request.args.get("has_annotations", None)
    has_annotations = None
    if has_annotations_raw == "true":
        has_annotations = True
    elif has_annotations_raw == "false":
        has_annotations = False

    active_filter = "all"
    if has_annotations is True:
        active_filter = "annotated"
    elif has_annotations is False:
        active_filter = "pending"

    result = get_project_files_paginated(
        project_id,
        page=page,
        per_page=per_page,
        split=split,
        has_annotations=has_annotations,
    )

    try:
        labels = json.loads(project["labels"]) or []
    except (json.JSONDecodeError, TypeError):
        labels = []

    return render_template(
        "project.html",
        app_name="Datamarkin",
        active_tab="project_detail",
        project=project,
        labels=labels,
        files=result["items"],
        pagination=result,
        active_filter=active_filter,
    )


ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def project_upload_route(project_id: str):
    project = get_project_by_id(project_id)
    if not project:
        abort(404)

    if 'file' not in request.files:
        abort(400)

    f = request.files['file']
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        abort(400)

    file_id = new_id()
    filename = f"{file_id}{ext}"
    dest = get_file_path(filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    f.save(str(dest))

    try:
        with Image.open(dest) as img:
            width, height = img.size
    except Exception:
        dest.unlink(missing_ok=True)
        abort(400)

    filesize = dest.stat().st_size
    insert_file(file_id, project_id, filename, ext, width, height, filesize)

    return jsonify({"id": file_id, "filename": filename, "width": width, "height": height}), 201


def project_image_page_route(project_id: str, file_id: str):
    project = get_project_by_id(project_id)
    if not project:
        abort(404)

    files = get_project_files(project_id)

    current_file = get_file_by_id(file_id)
    if not current_file:
        abort(404)

    current_index = next(
        (i for i, f in enumerate(files) if f["id"] == file_id), 0
    )

    try:
        labels = json.loads(project["labels"]) or []
    except (json.JSONDecodeError, TypeError):
        labels = []

    try:
        annotations = json.loads(current_file["annotations"]) if current_file["annotations"] else None
    except (json.JSONDecodeError, TypeError):
        annotations = None

    return render_template(
        "project_image.html",
        project=project,
        files=files,
        current_file=current_file,
        current_index=current_index,
        labels=labels,
        annotations=annotations,
    )
