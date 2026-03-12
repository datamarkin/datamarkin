#!/usr/bin/env python3
"""Seed database with sample projects and download images from picsum.photos."""

import argparse, json, math, os, random, shutil, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, DB_PATH, FILES_DIR, file_path
from db import init_db, get_db, now, new_id

PICSUM_LIST_URLS = [
    f"https://picsum.photos/v2/list?page={p}&limit=100" for p in range(1, 4)
]

PROJECT_TYPES = ["object_detection", "classification", "segmentation", "keypoints"]

ANNOTATION_RATE = 0.7

DISTINCT_COLORS = [
    "e6194b", "3cb44b", "ffe119", "4363d8", "f58231",
    "911eb4", "42d4f4", "f032e6", "bfef45", "fabed4",
    "469990", "dcbeff", "9a6324", "fffac8", "800000",
    "aaffc3", "808000", "ffd8b1", "000075", "a9a9a9",
]

LABEL_NAME_POOLS = {
    "object_detection": [
        "car", "person", "truck", "bicycle", "motorcycle", "bus", "traffic light",
        "stop sign", "fire hydrant", "bench", "bird", "cat", "dog", "horse",
        "backpack", "umbrella", "handbag", "suitcase", "bottle",
    ],
    "classification": [
        "normal", "defective", "cracked", "scratched", "dented", "rusted",
        "chipped", "stained", "warped", "corroded", "pristine", "faded",
        "bent", "torn", "discolored", "peeling", "bulging", "pitted",
    ],
    "segmentation": [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "vegetation", "terrain", "sky", "water", "ground", "bridge",
        "rail", "truck", "car", "bus", "motorcycle", "bicycle", "person",
    ],
    "keypoints": [
        "person", "dog", "cat", "horse", "bird", "hand", "face",
        "fish", "lizard", "rabbit", "cow", "sheep", "elephant",
        "giraffe", "bear", "deer", "frog",
    ],
}

KEYPOINT_TEMPLATES = {
    "person": {
        "keypoints": [
            {"id": 0, "name": "head", "color": "e6194b"},
            {"id": 1, "name": "neck", "color": "3cb44b"},
            {"id": 2, "name": "right shoulder", "color": "ffe119"},
            {"id": 3, "name": "right elbow", "color": "4363d8"},
            {"id": 4, "name": "right wrist", "color": "f58231"},
            {"id": 5, "name": "left shoulder", "color": "911eb4"},
            {"id": 6, "name": "left elbow", "color": "42d4f4"},
            {"id": 7, "name": "left wrist", "color": "f032e6"},
            {"id": 8, "name": "right hip", "color": "bfef45"},
            {"id": 9, "name": "left hip", "color": "fabed4"},
            {"id": 10, "name": "right knee", "color": "469990"},
        ],
        "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [1, 9], [8, 10]],
    },
    "dog": {
        "keypoints": [
            {"id": 0, "name": "nose", "color": "e6194b"},
            {"id": 1, "name": "head", "color": "3cb44b"},
            {"id": 2, "name": "neck", "color": "ffe119"},
            {"id": 3, "name": "front left paw", "color": "4363d8"},
            {"id": 4, "name": "front right paw", "color": "f58231"},
            {"id": 5, "name": "tail", "color": "911eb4"},
        ],
        "skeleton": [[0, 1], [1, 2], [2, 3], [2, 4], [2, 5]],
    },
    "cat": {
        "keypoints": [
            {"id": 0, "name": "nose", "color": "e6194b"},
            {"id": 1, "name": "left ear", "color": "3cb44b"},
            {"id": 2, "name": "right ear", "color": "ffe119"},
            {"id": 3, "name": "neck", "color": "4363d8"},
            {"id": 4, "name": "front left paw", "color": "f58231"},
            {"id": 5, "name": "front right paw", "color": "911eb4"},
            {"id": 6, "name": "tail tip", "color": "42d4f4"},
        ],
        "skeleton": [[0, 1], [0, 2], [0, 3], [3, 4], [3, 5], [3, 6]],
    },
    "horse": {
        "keypoints": [
            {"id": 0, "name": "nose", "color": "e6194b"},
            {"id": 1, "name": "head", "color": "3cb44b"},
            {"id": 2, "name": "neck", "color": "ffe119"},
            {"id": 3, "name": "withers", "color": "4363d8"},
            {"id": 4, "name": "front left hoof", "color": "f58231"},
            {"id": 5, "name": "front right hoof", "color": "911eb4"},
            {"id": 6, "name": "rear left hoof", "color": "42d4f4"},
            {"id": 7, "name": "rear right hoof", "color": "f032e6"},
        ],
        "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [3, 6], [3, 7]],
    },
    "bird": {
        "keypoints": [
            {"id": 0, "name": "beak", "color": "e6194b"},
            {"id": 1, "name": "head", "color": "3cb44b"},
            {"id": 2, "name": "left wing", "color": "ffe119"},
            {"id": 3, "name": "right wing", "color": "4363d8"},
            {"id": 4, "name": "tail", "color": "f58231"},
        ],
        "skeleton": [[0, 1], [1, 2], [1, 3], [1, 4]],
    },
    "hand": {
        "keypoints": [
            {"id": 0, "name": "wrist", "color": "e6194b"},
            {"id": 1, "name": "thumb tip", "color": "3cb44b"},
            {"id": 2, "name": "index tip", "color": "ffe119"},
            {"id": 3, "name": "middle tip", "color": "4363d8"},
            {"id": 4, "name": "ring tip", "color": "f58231"},
            {"id": 5, "name": "pinky tip", "color": "911eb4"},
        ],
        "skeleton": [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
    },
    "face": {
        "keypoints": [
            {"id": 0, "name": "left eye", "color": "e6194b"},
            {"id": 1, "name": "right eye", "color": "3cb44b"},
            {"id": 2, "name": "nose", "color": "ffe119"},
            {"id": 3, "name": "left mouth", "color": "4363d8"},
            {"id": 4, "name": "right mouth", "color": "f58231"},
        ],
        "skeleton": [[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4]],
    },
}

# Generic fallback for keypoint labels without a specific template
GENERIC_KEYPOINT_TEMPLATE = {
    "keypoints": [
        {"id": 0, "name": "point 1", "color": "e6194b"},
        {"id": 1, "name": "point 2", "color": "3cb44b"},
        {"id": 2, "name": "point 3", "color": "ffe119"},
        {"id": 3, "name": "point 4", "color": "4363d8"},
        {"id": 4, "name": "point 5", "color": "f58231"},
    ],
    "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4]],
}

PROJECTS = [
    {
        "name": "Object Detection - Urban Scenes",
        "status": "active",
        "train": 0,
        "model_architecture": "detectron2",
        "description": "Training dataset for detecting vehicles, pedestrians, and traffic signs in city environments.",
        "configuration": '{"batch_size": 32, "epochs": 100, "lr": 0.001}',
        "augmentation": '{"horizontal_flip": true, "resize": 640, "mosaic": true}',
        "preprocessing": '{"normalize": true, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}',
        "image_count": (50, 70),
        "splits": ["train", "train", "train", "val"],
    },
    {
        "name": "Medical Imaging - X-Ray",
        "status": "active",
        "train": 1,
        "model_architecture": "ResNet50",
        "description": "Chest X-ray images annotated for pneumonia and fracture detection.",
        "configuration": '{"batch_size": 16, "epochs": 50, "lr": 0.0001}',
        "augmentation": '{"random_rotation": 15, "random_crop": 0.1}',
        "preprocessing": '{"normalize": true, "mean": [127.5], "std": [127.5]}',
        "image_count": (40, 60),
        "splits": ["train", "train", "train", "test"],
    },
    {
        "name": "Satellite Imagery Analysis",
        "status": "training",
        "train": 1,
        "model_architecture": "U-Net",
        "description": "Land use classification for agricultural monitoring and urban planning.",
        "configuration": '{"batch_size": 8, "epochs": 200, "lr": 0.00001}',
        "augmentation": '{"color_jitter": 0.2, "random_rotation": 5}',
        "preprocessing": '{"normalize": true, "mean": [0.5], "std": [0.5]}',
        "image_count": (30, 50),
        "splits": ["train", "train", "val"],
    },
    {
        "name": "Old Facial Recognition",
        "status": "archived",
        "train": 0,
        "model_architecture": "FaceNet",
        "description": "Archived project for facial recognition - superseded by newer model.",
        "configuration": '{"batch_size": 32, "epochs": 100}',
        "augmentation": '{"horizontal_flip": true}',
        "preprocessing": '{"normalize": true}',
        "image_count": (30, 40),
        "splits": ["train", "train", "train", "val"],
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def generate_labels(project_type, count):
    """Generate a list of labels for a project type."""
    pool = LABEL_NAME_POOLS[project_type]
    names = random.sample(pool, min(count, len(pool)))
    colors = random.sample(DISTINCT_COLORS, min(count, len(DISTINCT_COLORS)))
    labels = []
    for i, name in enumerate(names):
        label = {"id": i, "name": name, "color": colors[i]}
        if project_type == "keypoints":
            tmpl = KEYPOINT_TEMPLATES.get(name, GENERIC_KEYPOINT_TEMPLATE)
            label["keypoints"] = tmpl["keypoints"]
            label["skeleton"] = tmpl["skeleton"]
        labels.append(label)
    return labels


def generate_bbox():
    """Generate a random normalized bounding box [x_min, y_min, x_max, y_max]."""
    w = random.uniform(0.05, 0.5)
    h = random.uniform(0.05, 0.5)
    x_min = random.uniform(0, 1 - w)
    y_min = random.uniform(0, 1 - h)
    x_max = round(x_min + w, 4)
    y_max = round(y_min + h, 4)
    return [round(x_min, 4), round(y_min, 4), x_max, y_max]


def generate_polygon(bbox):
    """Generate a convex polygon within a bounding box as flat [x1,y1,x2,y2,...]."""
    x_min, y_min, x_max, y_max = bbox
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    num_points = random.randint(6, 12)

    points = []
    for _ in range(num_points):
        px = random.gauss(cx, (x_max - x_min) / 4)
        py = random.gauss(cy, (y_max - y_min) / 4)
        # Clamp within bbox
        px = max(x_min, min(x_max, px))
        py = max(y_min, min(y_max, py))
        points.append((px, py))

    # Sort by angle from center to form convex shape
    points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    flat = []
    for px, py in points:
        flat.extend([round(px, 4), round(py, 4)])
    return flat


def generate_keypoints_for_bbox(bbox, keypoint_defs):
    """Generate keypoints distributed vertically through a bounding box."""
    x_min, y_min, x_max, y_max = bbox
    n = len(keypoint_defs)
    result = []
    for i, kp_def in enumerate(keypoint_defs):
        # Distribute vertically with slight randomness
        frac = (i + 0.5) / n
        ky = y_min + (y_max - y_min) * frac + random.gauss(0, (y_max - y_min) * 0.05)
        kx = x_min + (x_max - x_min) * random.uniform(0.2, 0.8)
        # Clamp within bbox
        kx = max(x_min, min(x_max, kx))
        ky = max(y_min, min(y_max, ky))
        result.append({"id": kp_def["id"], "point": [round(kx, 4), round(ky, 4)]})
    return result


def generate_annotation(project_type, labels_list):
    """Generate a random annotation for the given project type and labels."""
    n_labels = len(labels_list)

    if project_type == "classification":
        return {"class": random.randint(0, n_labels - 1)}

    num_objects = random.randint(1, 10)
    objects = []

    for _ in range(num_objects):
        cls = random.randint(0, n_labels - 1)
        bbox = generate_bbox()

        if project_type == "object_detection":
            objects.append({"class": cls, "bbox": bbox})
        elif project_type == "segmentation":
            seg = generate_polygon(bbox)
            objects.append({"class": cls, "bbox": bbox, "segmentation": seg})
        elif project_type == "keypoints":
            label = labels_list[cls]
            kp_defs = label.get("keypoints", GENERIC_KEYPOINT_TEMPLATE["keypoints"])
            kps = generate_keypoints_for_bbox(bbox, kp_defs)
            objects.append({"class": cls, "bbox": bbox, "keypoints": kps})

    return {"objects": objects}


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def fetch_picsum_list():
    """Fetch image list from picsum API across multiple pages."""
    all_images = []
    for url in PICSUM_LIST_URLS:
        print(f"Fetching picsum image list from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "datamarkin-seed/1.0"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        all_images.extend(data)
        print(f"  Got {len(data)} images (total: {len(all_images)})")
    return all_images


def download_image(picsum_id, width, height, dest_path):
    """Download resized image from picsum and save to dest_path."""
    url = f"https://picsum.photos/id/{picsum_id}/{width}/{height}"
    req = urllib.request.Request(url, headers={"User-Agent": "datamarkin-seed/1.0"})
    with urllib.request.urlopen(req) as resp:
        with open(dest_path, "wb") as f:
            f.write(resp.read())


# ---------------------------------------------------------------------------
# Main seed logic
# ---------------------------------------------------------------------------

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
        if FILES_DIR.exists():
            shutil.rmtree(FILES_DIR)
            print(f"  Deleted {FILES_DIR}")

    init_db()

    picsum_images = fetch_picsum_list()
    conn = get_db()
    ts = now()
    img_idx = 0

    for proj_def in PROJECTS:
        proj_id = new_id()
        proj_type = random.choice(PROJECT_TYPES)
        label_count = random.randint(2, 10) if proj_type == "classification" else random.randint(1, 10)
        labels_list = generate_labels(proj_type, label_count)

        print(f"Creating project: {proj_def['name']} (type={proj_type}, labels={len(labels_list)}, id={proj_id})")

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
                proj_type,
                proj_def["train"],
                proj_def["model_architecture"],
                proj_def["description"],
                proj_def["configuration"],
                proj_def["augmentation"],
                proj_def["preprocessing"],
                json.dumps(labels_list),
            ),
        )

        count = random.randint(*proj_def["image_count"])
        print(f"  Downloading {count} images...")
        annotated_count = 0
        for i in range(count):
            file_id = new_id()
            pic = picsum_images[img_idx % len(picsum_images)]
            img_idx += 1
            split = random.choice(proj_def["splits"])
            w, h = int(pic["width"]), int(pic["height"])

            dest = file_path(f"{file_id}.jpg")
            dest.parent.mkdir(parents=True, exist_ok=True)

            print(f"  [{i+1}/{count}] Downloading picsum #{pic['id']} ({w}x{h}) -> {dest.name}")
            download_image(pic["id"], w, h, dest)
            filesize = dest.stat().st_size

            if random.random() < ANNOTATION_RATE:
                annotations = json.dumps(generate_annotation(proj_type, labels_list))
                annotated_count += 1
            else:
                annotations = None

            conn.execute(
                """INSERT INTO files
                   (id, project_id, filename, extension, width, height,
                    filesize, split, annotations, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id,
                    proj_id,
                    f"{file_id}.jpg",
                    ".jpg",
                    w,
                    h,
                    filesize,
                    split,
                    annotations,
                    ts,
                    ts,
                ),
            )

        print(f"  Annotated: {annotated_count}/{count} ({100*annotated_count//count}%)")

    conn.commit()
    conn.close()
    print("Done! Seeding complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the Datamarkin database")
    parser.add_argument("--fresh", action="store_true", help="Delete existing DB and project folders before seeding")
    args = parser.parse_args()
    seed(fresh=args.fresh)
