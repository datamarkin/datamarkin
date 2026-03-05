from flask import render_template, abort

from queries import get_all_projects, get_project_by_id, get_project_files


def projects_page_route(app_name):
    return render_template(
        "projects.html",
        app_name="Datamarkin",
        active_tab="projects",
        projects=get_all_projects(),
    )


def project_detail_route(project_id: str):
    project = get_project_by_id(project_id)
    if not project:
        abort(404)

    files = get_project_files(project_id)

    return render_template(
        "project_detail.html",
        app_name="Datamarkin",
        active_tab="project_detail",
        project=project,
        files=files,
    )
