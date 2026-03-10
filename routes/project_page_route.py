import json
from flask import render_template, abort, request
from queries import get_project_by_id, get_project_files_paginated


class Project(dict):
    """Wrapper for project dict that parses JSON fields on first access."""

    def __init__(self, data: dict):
        super().__init__(data)
        self._parsed = {}

    def _get_json(self, key: str):
        if key not in self._parsed:
            raw = self.get(key) or "{}"
            try:
                self._parsed[key] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                self._parsed[key] = {}
        return self._parsed[key]

    @property
    def labels(self):
        return self._get_json("labels") or []

    @property
    def configuration(self):
        return self._get_json("configuration") or {}

    @property
    def augmentation(self):
        return self._get_json("augmentation") or {}

    @property
    def preprocessing(self):
        return self._get_json("preprocessing") or {}


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

    return render_template(
        "project.html",
        project=project,
        files=result["items"],
        pagination=result,
        active_filter=active_filter,
    )