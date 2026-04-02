import math

from db import get_db, new_id, now


def paginate_query(
    table: str,
    *,
    page: int = 1,
    per_page: int = 100,
    filters: dict | None = None,
    raw_filters: list[tuple[str, list]] | None = None,
    order_by: str = "created_at DESC",
) -> dict:
    page = max(1, page)
    per_page = max(1, min(200, per_page))

    where_parts = []
    params = []

    if filters:
        for col, val in filters.items():
            where_parts.append(f"{col} = ?")
            params.append(val)

    if raw_filters:
        for sql_fragment, fragment_params in raw_filters:
            where_parts.append(f"({sql_fragment})")
            params.extend(fragment_params)

    where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

    conn = get_db()
    total = conn.execute(
        f"SELECT COUNT(*) FROM {table}{where_clause}", params
    ).fetchone()[0]

    total_pages = max(1, math.ceil(total / per_page))
    page = min(page, total_pages)
    offset = (page - 1) * per_page

    rows = conn.execute(
        f"SELECT * FROM {table}{where_clause} ORDER BY {order_by} LIMIT ? OFFSET ?",
        params + [per_page, offset],
    ).fetchall()
    conn.close()

    return {
        "items": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


def get_project_files_paginated(
    project_id: str,
    *,
    page: int = 1,
    per_page: int = 100,
    split: str | None = None,
    has_annotations: bool | None = None,
) -> dict:
    filters = {"project_id": project_id}
    raw_filters = []

    if split is not None:
        filters["split"] = split

    if has_annotations is True:
        raw_filters.append((
            "annotations IS NOT NULL AND annotations != '' AND annotations != '[]' AND annotations != 'null'",
            [],
        ))
    elif has_annotations is False:
        raw_filters.append((
            "annotations IS NULL OR annotations = '' OR annotations = '[]' OR annotations = 'null'",
            [],
        ))

    return paginate_query(
        "files",
        page=page,
        per_page=per_page,
        filters=filters,
        raw_filters=raw_filters if raw_filters else None,
        order_by="sort_order, created_at DESC",
    )


def get_all_projects() -> list[dict]:
    conn = get_db()
    rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_project_by_id(project_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_project(name: str, project_type: str, labels: str | None = None) -> str:
    conn = get_db()
    project_id = new_id()
    ts = now()
    conn.execute(
        """INSERT INTO projects (id, name, type, labels, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (project_id, name, project_type, labels, ts, ts),
    )
    conn.commit()
    conn.close()
    return project_id


def get_file_by_id(file_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def insert_file(file_id, project_id, filename, extension, width, height, filesize):
    conn = get_db()
    ts = now()
    conn.execute(
        """INSERT INTO files
           (id, project_id, filename, extension, width, height,
            filesize, split, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (file_id, project_id, filename, extension, width, height,
         filesize, None, ts, ts),
    )
    conn.commit()
    conn.close()
    return file_id


def update_project_info(project_id: str, name: str, description: str, labels: list) -> None:
    import json
    conn = get_db()
    conn.execute(
        "UPDATE projects SET name = ?, description = ?, labels = ?, updated_at = ? WHERE id = ?",
        (name.strip(), description.strip(), json.dumps(labels), now(), project_id),
    )
    conn.commit()
    conn.close()


def update_project_pipeline(project_id: str, key: str, pipeline_json: dict) -> None:
    import json
    conn = get_db()
    conn.execute(
        f"UPDATE projects SET {key} = ?, updated_at = ? WHERE id = ?",
        (json.dumps(pipeline_json), now(), project_id),
    )
    conn.commit()
    conn.close()


def update_project_configuration(project_id: str, config_json: str) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE projects SET configuration = ?, updated_at = ? WHERE id = ?",
        (config_json, now(), project_id),
    )
    conn.commit()
    conn.close()


def assign_file_splits(project_id: str, train_ratio: float, val_ratio: float, test_ratio: float) -> dict:
    import random
    conn = get_db()
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM files WHERE project_id = ? ORDER BY created_at", (project_id,)
    ).fetchall()]
    random.shuffle(ids)
    n = len(ids)
    n_train = round(n * train_ratio)
    n_val   = round(n * val_ratio)
    splits = (
        [("train", i) for i in ids[:n_train]] +
        [("valid", i) for i in ids[n_train:n_train + n_val]] +
        [("test",  i) for i in ids[n_train + n_val:]]
    )
    ts = now()
    for split, file_id in splits:
        conn.execute("UPDATE files SET split = ?, updated_at = ? WHERE id = ?", (split, ts, file_id))
    conn.commit()
    conn.close()
    return {"train": n_train, "valid": n_val, "test": n - n_train - n_val}


def update_file_annotations(file_id: str, annotations_json: str) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE files SET annotations = ?, updated_at = ? WHERE id = ?",
        (annotations_json, now(), file_id),
    )
    conn.commit()
    conn.close()


def get_project_files(project_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM files WHERE project_id = ? ORDER BY sort_order, created_at DESC",
        (project_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ── Training queries ──────────────────────────────────────────────────────────

def create_training(project_id: str, config_json: str) -> str:
    conn = get_db()
    training_id = new_id()
    ts = now()
    conn.execute(
        """INSERT INTO trainings (id, project_id, status, config, created_at, updated_at)
           VALUES (?, ?, 'pending', ?, ?, ?)""",
        (training_id, project_id, config_json, ts, ts),
    )
    conn.commit()
    conn.close()
    return training_id


def get_training(training_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM trainings WHERE id = ?", (training_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_training_progress(training_id: str, progress_json: str) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE trainings SET progress = ?, updated_at = ? WHERE id = ?",
        (progress_json, now(), training_id),
    )
    conn.commit()
    conn.close()


def update_training_done(training_id: str, model_path: str, metrics_json: str) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE trainings SET status = 'done', model_path = ?, metrics = ?, updated_at = ? WHERE id = ?",
        (model_path, metrics_json, now(), training_id),
    )
    conn.commit()
    conn.close()


def update_training_status(training_id: str, status: str, error: str | None = None) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE trainings SET status = ?, error = ?, updated_at = ? WHERE id = ?",
        (status, error, now(), training_id),
    )
    conn.commit()
    conn.close()


def get_done_trainings() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM trainings WHERE status = 'done' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_project_trainings(project_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM trainings WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ── Workflow queries ───────────────────────────────────────────────────────────

def list_workflows() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM workflows ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_workflow_by_id(workflow_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_workflow(name: str, description: str, workflow_json: str) -> dict:
    conn = get_db()
    workflow_id = new_id()
    ts = now()
    conn.execute(
        """INSERT INTO workflows (id, name, description, workflow_json, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (workflow_id, name, description, workflow_json, ts, ts),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
    conn.close()
    return dict(row)


def update_workflow(workflow_id: str, data: dict) -> dict | None:
    allowed = {"name", "description", "workflow_json"}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return get_workflow_by_id(workflow_id)
    conn = get_db()
    ts = now()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    conn.execute(
        f"UPDATE workflows SET {set_clause}, updated_at = ? WHERE id = ?",
        [*updates.values(), ts, workflow_id],
    )
    conn.commit()
    row = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_workflow(workflow_id: str) -> None:
    conn = get_db()
    conn.execute("DELETE FROM workflows WHERE id = ?", (workflow_id,))
    conn.commit()
    conn.close()
