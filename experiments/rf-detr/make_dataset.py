"""
Generates a minimal synthetic COCO dataset for smoke-testing the RF-DETR
training pipeline. Produces random noise images with fake bounding boxes and
segmentation polygons — usable by both main.py (detection) and seg.py.

Usage:
    python make_dataset.py
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = {"train": 20, "valid": 5, "test": 5}
CATEGORIES = [
    {"id": 1, "name": "cat", "supercategory": "animal"},
    {"id": 2, "name": "dog", "supercategory": "animal"},
]
IMG_W, IMG_H = 1000, 1000

for split, n in SPLITS.items():
    split_dir = Path("dataset") / split
    split_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1

    for i in range(1, n + 1):
        fname = f"{split}_{i:04d}.jpg"
        pixels = np.random.randint(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(split_dir / fname)
        images.append({"id": i, "file_name": fname, "width": IMG_W, "height": IMG_H})

        for _ in range(2):
            x = random.randint(0, IMG_W - 20)
            y = random.randint(0, IMG_H - 20)
            w = random.randint(10, min(30, IMG_W - x))
            h = random.randint(10, min(30, IMG_H - y))
            # 4-point polygon (top-left → top-right → bottom-right → bottom-left)
            seg = [x, y, x + w, y, x + w, y + h, x, y + h]
            annotations.append({
                "id": ann_id,
                "image_id": i,
                "category_id": random.choice([1, 2]),
                "bbox": [x, y, w, h],
                "area": w * h,
                "segmentation": [seg],
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images, "categories": CATEGORIES, "annotations": annotations}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco, indent=2))
    print(f"{split}: {n} images → {split_dir}/")

print("Done. Run: python main.py  (detection)  or  python seg.py  (segmentation)")
