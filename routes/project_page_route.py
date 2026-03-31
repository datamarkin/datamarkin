import json

from flask import render_template, abort, request
from db_models import Project
from queries import get_project_by_id, get_project_files_paginated, get_project_trainings


def project_page_route(project_id: str):
    raw_project = get_project_by_id(project_id)
    if not raw_project:
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

    # Wrap in Project class for dot notation access (project.labels, etc.)
    project = Project(raw_project)

    # Parse training JSON fields so template can access them as dicts
    training_history = []
    for t in get_project_trainings(project_id):
        td = dict(t)
        td["config"] = json.loads(td.get("config") or "{}")
        td["metrics"] = json.loads(td.get("metrics") or "{}")
        td["progress"] = json.loads(td.get("progress") or "{}")
        training_history.append(td)

    active_training = next(
        (t for t in training_history if t["status"] in ("pending", "running")), None
    )

    return render_template(
        "project.html",
        project=project,
        files=result["items"],
        pagination=result,
        active_filter=active_filter,
        training_history=training_history,
        active_training=active_training,
    )