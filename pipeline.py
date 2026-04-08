"""
Pipeline conversion utilities.

Converts the SortableJS pipeline builder JSON (stored in
projects.preprocessing / projects.augmentation) to:
  - a runnable albumentations.Compose pipeline (for preview / export)
  - an RF-DETR aug_config dict (for training integration)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import albumentations as A

# ---------------------------------------------------------------------------
# Our JSON → albumentations.Compose
# ---------------------------------------------------------------------------

# Transforms where UI params need remapping before passing to albumentations
_REMAP = {
    "Normalize": lambda p: {
        "mean": [p.get("mean_r", 0.485), p.get("mean_g", 0.456), p.get("mean_b", 0.406)],
        "std": [p.get("std_r", 0.229), p.get("std_g", 0.224), p.get("std_b", 0.225)],
    },
    # JpegCompression was renamed ImageCompression in albumentations v2
    "JpegCompression": lambda p: {
        "quality_lower": int(p.get("quality_lower", 50)),
        "quality_upper": int(p.get("quality_upper", 99)),
    },
}

_TYPE_OVERRIDE = {
    "JpegCompression": "ImageCompression",
}


def _build_transform(t: dict, is_aug: bool = True) -> A.BasicTransform:
    """Convert a single transform entry from our JSON to an albumentations instance."""
    t_type = t["type"]
    raw_params = t.get("params", {})
    p = float(t.get("p", 1.0)) if is_aug else 1.0

    if t_type == "OneOf":
        children = [_build_transform(c, is_aug) for c in t.get("children", [])]
        return A.OneOf(children, p=p)

    albu_type = _TYPE_OVERRIDE.get(t_type, t_type)
    params = _REMAP[t_type](raw_params) if t_type in _REMAP else dict(raw_params)
    return getattr(A, albu_type)(p=p, **params)


def build_preprocessing_pipeline(pipeline_json: dict) -> A.Compose:
    """Convert stored pre-processing JSON to an albumentations Compose pipeline."""
    transforms = [
        _build_transform(t, is_aug=False)
        for t in pipeline_json.get("transforms", [])
    ]
    return A.Compose(transforms)


def build_augmentation_pipeline(pipeline_json: dict) -> A.Compose:
    """Convert stored augmentation JSON to an albumentations Compose pipeline."""
    transforms = [
        _build_transform(t, is_aug=True)
        for t in pipeline_json.get("transforms", [])
    ]
    return A.Compose(transforms)


# ---------------------------------------------------------------------------
# Our JSON → RF-DETR aug_config format
# ---------------------------------------------------------------------------
#
# RF-DETR's AlbumentationsWrapper.from_config() expects a list of single-key
# dicts:  [{"HorizontalFlip": {"p": 0.5}}, {"OneOf": {"transforms": [...]}}]
#
# Our UI stores:
#   {"transforms": [
#       {"type": "HorizontalFlip", "p": 0.5, "params": {}},
#       {"type": "OneOf", "p": 0.3, "children": [
#           {"type": "GaussianBlur", "p": 1.0, "params": {"blur_limit": 7}}
#       ]}
#   ]}


def _convert_transform_to_rfdetr(t: dict) -> Dict[str, Any]:
    """Convert a single UI transform entry to RF-DETR's {name: params} format."""
    t_type = t["type"]
    albu_type = _TYPE_OVERRIDE.get(t_type, t_type)
    raw_params = t.get("params", {})

    if t_type == "OneOf":
        children_rfdetr = [_convert_transform_to_rfdetr(c) for c in t.get("children", [])]
        return {albu_type: {"transforms": children_rfdetr, "p": float(t.get("p", 1.0))}}

    # Merge p and params into a single dict
    params = _REMAP[t_type](raw_params) if t_type in _REMAP else dict(raw_params)
    if "p" in t:
        params["p"] = float(t["p"])
    return {albu_type: params}


def to_rfdetr_aug_config(
    augmentation_json: Optional[dict],
) -> List[Dict[str, Any]]:
    """Convert our stored augmentation JSON to RF-DETR's aug_config list format.

    Returns an empty list when no augmentation is configured, which tells
    RF-DETR to skip all augmentations (including its built-in defaults).
    """
    if not augmentation_json:
        return []
    transforms = augmentation_json.get("transforms", [])
    if not transforms:
        return []
    return [_convert_transform_to_rfdetr(t) for t in transforms]
