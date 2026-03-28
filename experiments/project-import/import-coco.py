#!/usr/bin/env python3
"""Import a COCO-format dataset into the Datamarkin database and file store."""

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Hard-coded parameters — edit these before running
# ---------------------------------------------------------------------------

PROJECT_NAME = "COCO-50 Test Import seg 100"
NUM_IMAGES   = 100          # set to None to import all images
SOURCE_DIR   = Path("/XXX/test")
PROJECT_TYPE = "segmentation"  # "segmentation" or "object_detection" to also store mask polygons

# ---------------------------------------------------------------------------
# Bootstrap: add project root to path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import file_path   # noqa: E402
from db import init_db, get_db, now, new_id  # noqa: E402

# ---------------------------------------------------------------------------
# Colours (same list as scripts/seed.py)
# ---------------------------------------------------------------------------

DISTINCT_COLORS = [
    "e6194b", "3cb44b", "ffe119", "4363d8", "f58231",
    "911eb4", "42d4f4", "f032e6", "bfef45", "fabed4",
    "469990", "dcbeff", "9a6324", "fffac8", "800000",
    "aaffc3", "808000", "ffd8b1", "000075", "a9a9a9",
]

# ---------------------------------------------------------------------------
# Main import logic
# ---------------------------------------------------------------------------

def main():
    annotations_file = SOURCE_DIR / "_annotations.coco.json"
    if not annotations_file.exists():
        sys.exit(f"ERROR: annotations file not found: {annotations_file}")

    print(f"Loading {annotations_file} ...")
    with open(annotations_file) as f:
        coco = json.load(f)

    # --- Build category mapping -------------------------------------------
    # Skip category id 0 if its name is "object" (generic supercategory)
    categories = [
        c for c in coco["categories"]
        if not (c["id"] == 0 and c["name"] == "object")
    ]
    # Map coco cat id -> sequential project label id (0-indexed)
    cat_id_to_label_id = {c["id"]: i for i, c in enumerate(categories)}
    labels = [
        {
            "id": i,
            "name": c["name"],
            "color": DISTINCT_COLORS[i % len(DISTINCT_COLORS)],
        }
        for i, c in enumerate(categories)
    ]
    print(f"Categories: {len(labels)}")

    # --- Build image / annotation lookup dicts ----------------------------
    anns_by_image_id = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image_id[ann["image_id"]].append(ann)

    # --- Select images to import ------------------------------------------
    all_images = coco["images"]
    selected = random.sample(all_images, min(NUM_IMAGES, len(all_images))) if NUM_IMAGES is not None else all_images
    print(f"Images to import: {len(selected)}")

    # --- Init DB ----------------------------------------------------------
    init_db()
    conn = get_db()
    ts = now()

    # --- Create project ---------------------------------------------------
    proj_id = new_id()
    conn.execute(
        """INSERT INTO projects
           (id, name, status, sort_order, created_at, updated_at,
            type, train, labels)
           VALUES (?, ?, 'active', 0, ?, ?, ?, 0, ?)""",
        (proj_id, PROJECT_NAME, ts, ts, PROJECT_TYPE, json.dumps(labels)),
    )
    print(f"Created project: {proj_id}  ({PROJECT_NAME})")

    # --- Import images ----------------------------------------------------
    imported = 0
    ann_objects = 0
    annotated_files = 0
    coco_id_to_file_id = {}

    for i, img in enumerate(selected):
        src = SOURCE_DIR / img["file_name"]
        if not src.exists():
            print(f"  [SKIP] not found: {src.name}")
            continue

        ext = src.suffix.lower()
        file_id = new_id()
        dest = file_path(f"{file_id}{ext}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        filesize = dest.stat().st_size

        w = img["width"]
        h = img["height"]

        conn.execute(
            """INSERT INTO files
               (id, project_id, filename, extension, width, height,
                filesize, split, annotations, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)""",
            (file_id, proj_id, f"{file_id}{ext}", ext, w, h, filesize, ts, ts),
        )

        coco_id_to_file_id[img["id"]] = (file_id, w, h)
        imported += 1
        print(f"  [{i+1}/{len(selected)}] {src.name} -> {file_id}{ext}")

    # --- Convert and save annotations ------------------------------------
    for coco_img_id, (file_id, W, H) in coco_id_to_file_id.items():
        anns = anns_by_image_id.get(coco_img_id, [])
        if not anns:
            continue

        objects = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_label_id:
                continue
            label_id = cat_id_to_label_id[cat_id]

            # Normalize bbox [x, y, w, h] -> [x_min, y_min, x_max, y_max] (normalized)
            bx, by, bw, bh = ann["bbox"]
            bbox = [
                round(bx / W, 6),
                round(by / H, 6),
                round((bx + bw) / W, 6),
                round((by + bh) / H, 6),
            ]

            if PROJECT_TYPE == "segmentation" and ann.get("segmentation"):
                # Use first polygon; segmentation is [[x1,y1,x2,y2,...]]
                poly = ann["segmentation"][0]
                seg = []
                for j in range(0, len(poly) - 1, 2):
                    seg.append(round(poly[j] / W, 6))
                    seg.append(round(poly[j + 1] / H, 6))
                objects.append({"class": label_id, "bbox": bbox, "segmentation": seg})
            else:
                objects.append({"class": label_id, "bbox": bbox})

        if objects:
            annotations_json = json.dumps({"objects": objects})
            conn.execute(
                "UPDATE files SET annotations=?, updated_at=? WHERE id=?",
                (annotations_json, ts, file_id),
            )
            ann_objects += len(objects)
            annotated_files += 1

    conn.commit()
    conn.close()

    print()
    print(f"Project:     {proj_id}  ({PROJECT_NAME})")
    print(f"Images:      {imported} imported")
    print(f"Annotations: {ann_objects} objects across {annotated_files} files")


if __name__ == "__main__":
    main()
