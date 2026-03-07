from db import get_db, new_id, now


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
