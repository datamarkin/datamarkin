import json
from flask import render_template, abort, request
from queries import get_project_by_id, get_project_files_paginated

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