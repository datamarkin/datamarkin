import os
from pathlib import Path

from flask import render_template, abort, request, redirect, url_for, jsonify
from PIL import Image

from config import file_path as get_file_path, ALLOWED_EXTENSIONS
from db import new_id
from db_models import Project, File
from queries import get_all_projects, get_project_by_id, get_project_files, get_file_by_id, create_project, insert_file, update_project_info, update_project_pipeline


def projects_page_route():
    return render_template(
        "projects.html",
        projects=[Project(p) for p in get_all_projects()],
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
    )


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


def project_pipeline_route(project_id: str):
    project = get_project_by_id(project_id)
    if not project:
        abort(404)
    data = request.get_json()
    key = data.get('key')
    if key not in ('preprocessing', 'augmentation'):
        return jsonify({'error': 'invalid key'}), 400
    update_project_pipeline(project_id, key, data.get('pipeline', {}))
    return jsonify({'ok': True})


def project_settings_route(project_id: str):
    project = get_project_by_id(project_id)
    if not project:
        abort(404)
    data = request.get_json()
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'name required'}), 400
    description = data.get('description', '') or ''
    labels = data.get('labels', [])
    update_project_info(project_id, name, description, labels)
    return jsonify({'ok': True, 'name': name})


def project_image_page_route(project_id: str, file_id: str):
    raw_project = get_project_by_id(project_id)
    if not raw_project:
        abort(404)

    files = get_project_files(project_id)

    current_file_raw = get_file_by_id(file_id)
    if not current_file_raw:
        abort(404)

    current_index = next(
        (i for i, f in enumerate(files) if f["id"] == file_id), 0
    )

    wrapped_project = Project(raw_project)
    wrapped_file = File(current_file_raw) if current_file_raw else None

    return render_template(
        "project_image.html",
        project=wrapped_project,
        files=files,
        current_file=wrapped_file,
        current_index=current_index,
    )
