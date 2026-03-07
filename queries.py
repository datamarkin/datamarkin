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
            filesize, is_annotated, split, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (file_id, project_id, filename, extension, width, height,
         filesize, 0, None, ts, ts),
    )
    conn.commit()
    conn.close()
    return file_id


def get_project_files(project_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM files WHERE project_id = ? ORDER BY sort_order, created_at DESC",
        (project_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
