"""Annotation deduplication utilities."""


def bbox_iou(a, b):
    """IoU between two [x_min, y_min, x_max, y_max] bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def deduplicate_objects(new_objects, existing_objects, iou_threshold=0.9):
    """Return only new_objects that don't overlap existing ones above threshold."""
    kept = []
    for new_obj in new_objects:
        if not new_obj.get("bbox"):
            kept.append(new_obj)
            continue
        is_dup = any(
            existing.get("bbox") and bbox_iou(new_obj["bbox"], existing["bbox"]) >= iou_threshold
            for existing in existing_objects
        )
        if not is_dup:
            kept.append(new_obj)
    return kept
