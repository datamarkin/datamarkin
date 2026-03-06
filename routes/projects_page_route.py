import json

from flask import render_template, abort

from queries import get_all_projects, get_project_by_id, get_project_files, get_file_by_id


def projects_page_route(app_name):
    return render_template(
        "projects.html",
        app_name="Datamarkin",
        active_tab="projects",
        projects=get_all_projects(),
    )


def project_page_route(project_id: str):
    project = get_project_by_id(project_id)
    if not project:
        abort(404)

    files = get_project_files(project_id)

    return render_template(
        "project.html",
        app_name="Datamarkin",
        active_tab="project_detail",
        project=project,
        files=files,
    )


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
        labels = json.loads(project["labels"]) or {}
    except (json.JSONDecodeError, TypeError):
        labels = {}

    return render_template(
        "project_image.html",
        project=project,
        files=files,
        current_file=current_file,
        current_index=current_index,
        labels=labels,
    )
