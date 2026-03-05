#!/usr/bin/env python3
"""Seed database with sample projects and download images from picsum.photos."""

import argparse, json, os, shutil, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, DB_PATH
from db import init_db, get_db, now, new_id

PICSUM_LIST_URL = "https://picsum.photos/v2/list?page=2&limit=100"

PROJECTS = [
    {
        "name": "Object Detection - Urban Scenes",
        "status": "active",
        "type": "object_detection",
        "train": 0,
        "model_architecture": "detectron2",
        "description": "Training dataset for detecting vehicles, pedestrians, and traffic signs in city environments.",
        "configuration": '{"batch_size": 32, "epochs": 100, "lr": 0.001}',
        "augmentation": '{"horizontal_flip": true, "resize": 640, "mosaic": true}',
        "preprocessing": '{"normalize": true, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}',
        "labels": '{"car": "#ff0000", "person": "#00ff00", "traffic sign": "#ffff00", "bike": "#0000ff"}',
        "images": [
            {"w": 640, "h": 480, "split": "train"},
            {"w": 640, "h": 480, "split": "train"},
            {"w": 640, "h": 480, "split": "val"},
        ],
    },
    {
        "name": "Medical Imaging - X-Ray",
        "status": "active",
        "type": "classification",
        "train": 1,
        "model_architecture": "ResNet50",
        "description": "Chest X-ray images annotated for pneumonia and fracture detection.",
        "configuration": '{"batch_size": 16, "epochs": 50, "lr": 0.0001}',
        "augmentation": '{"random_rotation": 15, "random_crop": 0.1}',
        "preprocessing": '{"normalize": true, "mean": [127.5], "std": [127.5]}',
        "labels": '{"pneumonia": "#ff0000", "fracture": "#ffaa00", "normal": "#00ff00"}',
        "images": [
            {"w": 512, "h": 512, "split": "train"},
            {"w": 512, "h": 512, "split": "test"},
        ],
    },
    {
        "name": "Satellite Imagery Analysis",
        "status": "training",
        "type": "segmentation",
        "train": 1,
        "model_architecture": "U-Net",
        "description": "Land use classification for agricultural monitoring and urban planning.",
        "configuration": '{"batch_size": 8, "epochs": 200, "lr": 0.00001}',
        "augmentation": '{"color_jitter": 0.2, "random_rotation": 5}',
        "preprocessing": '{"normalize": true, "mean": [0.5], "std": [0.5]}',
        "labels": '{"forest": "#228B22", "agriculture": "#90EE90", "urban": "#808080", "water": "#4169E1"}',
        "images": [
            {"w": 800, "h": 600, "split": "train"},
        ],
    },
    {
        "name": "Old Facial Recognition",
        "status": "archived",
        "type": "classification",
        "train": 0,
        "model_architecture": "FaceNet",
        "description": "Archived project for facial recognition - superseded by newer model.",
        "configuration": '{"batch_size": 32, "epochs": 100}',
        "augmentation": '{"horizontal_flip": true}',
        "preprocessing": '{"normalize": true}',
        "labels": '{"face": "#ffffff"}',
        "images": [],
    },
]


def fetch_picsum_list():
    """Fetch image list from picsum API and return as list of dicts."""
    print("Fetching picsum image list...")
    req = urllib.request.Request(PICSUM_LIST_URL, headers={"User-Agent": "datamarkin-seed/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    print(f"  Got {len(data)} images from picsum")
    return data


def download_image(picsum_id, width, height, dest_path):
    """Download resized image from picsum and save to dest_path."""
    url = f"https://picsum.photos/id/{picsum_id}/{width}/{height}"
    req = urllib.request.Request(url, headers={"User-Agent": "datamarkin-seed/1.0"})
    with urllib.request.urlopen(req) as resp:
        with open(dest_path, "wb") as f:
            f.write(resp.read())


def seed(fresh=False):
    if fresh:
        print("Fresh mode: removing existing data...")
        if DB_PATH.exists():
            DB_PATH.unlink()
            print(f"  Deleted {DB_PATH}")
        projects_dir = DATA_DIR / "projects"
        if projects_dir.exists():
            shutil.rmtree(projects_dir)
            print(f"  Deleted {projects_dir}")

    init_db()

    picsum_images = fetch_picsum_list()
    conn = get_db()
    ts = now()
    img_idx = 0

    for proj_def in PROJECTS:
        proj_id = new_id()
        print(f"Creating project: {proj_def['name']} ({proj_id})")

        conn.execute(
            """INSERT INTO projects
               (id, name, status, sort_order, created_at, updated_at,
                type, train, model_architecture, description,
                configuration, augmentation, preprocessing, labels)
               VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                proj_id,
                proj_def["name"],
                proj_def["status"],
                ts,
                ts,
                proj_def["type"],
                proj_def["train"],
                proj_def["model_architecture"],
                proj_def["description"],
                proj_def["configuration"],
                proj_def["augmentation"],
                proj_def["preprocessing"],
                proj_def["labels"],
            ),
        )

        for img_def in proj_def["images"]:
            file_id = new_id()
            picsum_id = picsum_images[img_idx]["id"]
            img_idx += 1

            dest_dir = DATA_DIR / "projects" / proj_id / "images"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{file_id}.jpg"

            print(f"  Downloading picsum #{picsum_id} ({img_def['w']}x{img_def['h']}) -> {dest.name}")
            download_image(picsum_id, img_def["w"], img_def["h"], dest)
            filesize = dest.stat().st_size

            conn.execute(
                """INSERT INTO files
                   (id, project_id, filename, extension, width, height,
                    filesize, is_annotated, split, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id,
                    proj_id,
                    f"{file_id}.jpg",
                    ".jpg",
                    img_def["w"],
                    img_def["h"],
                    filesize,
                    0,
                    img_def["split"],
                    ts,
                    ts,
                ),
            )

    conn.commit()
    conn.close()
    print("Done! Seeding complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the Datamarkin database")
    parser.add_argument("--fresh", action="store_true", help="Delete existing DB and project folders before seeding")
    args = parser.parse_args()
    seed(fresh=args.fresh)
