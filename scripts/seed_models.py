"""Seed built-in detectron2 models as training records.

Usage:
    python -m scripts.seed_models          # add models (skip existing)
    python -m scripts.seed_models --fresh  # delete and re-create

Each model gets a fixed UUID so re-running the script is idempotent.
Model weights are NOT downloaded here — they download from dtmfiles.com
on first inference via pf.assets.download().
"""

import json
import sys

from db import get_db, init_db, now

# ── Built-in detectron2 models ──────────────────────────────────────────────
# Fill in your actual models below. Each entry becomes a training record
# with status="done" that appears in Studio.
#
# Required fields:
#   id            — fixed UUID (keeps re-seeding idempotent)
#   name          — display name shown in Studio
#   variant       — detectron2 architecture (e.g. "mask_rcnn_R_50_FPN_3x")
#   model_path    — dtmfiles asset path (e.g. "detectron2/my_model.pth")
#   project_type  — "detection", "segmentation", or "keypoint_detection"
#   labels        — list of {"id": int, "name": str, "color": "#hex"}
#                   for keypoint models, include "keypoints" and "skeleton"

BUILTIN_MODELS = [
    # Example — replace with your actual models:
    #
    # {
    #     "id": "builtin-d2-mask-rcnn-cells",
    #     "name": "Cell Instance Segmentation",
    #     "variant": "mask_rcnn_R_50_FPN_3x",
    #     "model_path": "detectron2/cell_mask_rcnn_R_50_FPN.pth",
    #     "project_type": "segmentation",
    #     "labels": [
    #         {"id": 0, "name": "cell", "color": "#FF6B6B"},
    #         {"id": 1, "name": "nucleus", "color": "#4ECDC4"},
    #     ],
    # },
    #
    # Keypoint model example:
    # {
    #     "id": "builtin-d2-keypoint-skeleton",
    #     "name": "Skeleton Keypoint Detection",
    #     "variant": "keypoint_rcnn_R_50_FPN_3x",
    #     "model_path": "detectron2/skeleton_keypoint_rcnn.pth",
    #     "project_type": "keypoint_detection",
    #     "labels": [
    #         {
    #             "id": 0, "name": "person", "color": "#FF6B6B",
    #             "keypoints": [
    #                 {"id": 0, "name": "head", "color": "#FF0000"},
    #                 {"id": 1, "name": "left_shoulder", "color": "#00FF00"},
    #                 {"id": 2, "name": "right_shoulder", "color": "#0000FF"},
    #             ],
    #             "skeleton": [[0, 1], [0, 2]],
    #         },
    #     ],
    # },
]


def seed_builtin_models(fresh=False):
    if not BUILTIN_MODELS:
        print("No built-in models defined in BUILTIN_MODELS. Add your models and re-run.")
        return

    init_db()
    conn = get_db()
    ts = now()

    if fresh:
        ids = [m["id"] for m in BUILTIN_MODELS]
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM trainings WHERE id IN ({placeholders})", ids)
        conn.commit()
        print(f"Cleared {len(ids)} built-in model records.")

    inserted = 0
    for m in BUILTIN_MODELS:
        existing = conn.execute(
            "SELECT id FROM trainings WHERE id = ?", (m["id"],)
        ).fetchone()
        if existing:
            print(f"  Skip (exists): {m['name']}")
            continue

        config = json.dumps({
            "name": m["name"],
            "model_architecture": "detectron2",
            "variant": m["variant"],
            "project_type": m["project_type"],
            "labels": m["labels"],
        })

        conn.execute(
            """INSERT INTO trainings
               (id, project_id, status, config, model_path, metrics, created_at, updated_at)
               VALUES (?, NULL, 'done', ?, ?, '{}', ?, ?)""",
            (m["id"], config, m["model_path"], ts, ts),
        )
        inserted += 1
        print(f"  Added: {m['name']}")

    conn.commit()
    conn.close()
    print(f"Done. Inserted {inserted} model(s).")


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    seed_builtin_models(fresh=fresh)
