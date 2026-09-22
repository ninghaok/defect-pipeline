"""Report-only ROI and foreground segmentation metrics (never training labels)."""
from pathlib import Path

import numpy as np
import cv2

from detected_pipeline.masks import external_gt, internal_mask


def segmentation_row(label, gt_path, prediction, shape, roi_path=None, external=True):
    """Raw mask counts; missing annotations stay unavailable, not model errors."""
    row = {"label": label, "iou": None, "gt_status": "not_required"}
    if label != "NG":
        return row
    if not gt_path or not Path(gt_path).is_file():
        row.update(gt_status="missing_gt", label="INVALID_GT" if roi_path else label)
        return row
    try:
        gt = (external_gt if external else internal_mask)(gt_path)
    except (ValueError, OSError, cv2.error):
        row.update(gt_status="unreadable_gt", label="INVALID_GT" if roi_path else label)
        return row
    if gt.shape != tuple(shape) or not gt.any():
        row.update(gt_status="invalid_gt", label="INVALID_GT" if roi_path else label)
        return row
    pred = np.zeros(gt.shape, bool) if prediction is None else np.asarray(prediction, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/image shape mismatch: {pred.shape} != {gt.shape}")
    if roi_path:
        roi = internal_mask(roi_path, gt.shape)
        if not roi.any():
            raise ValueError(f"Empty inspection ROI: {roi_path}")
        gt = gt & roi
        pred = pred & roi
    if not gt.any():
        row.update(label="EXCLUDED", gt_status="outside_roi")
        return row
    intersection = int((gt & pred).sum())
    union = int((gt | pred).sum())
    row.update(gt_status="valid", intersection=intersection, union=union,
               gt_pixels=int(gt.sum()), predicted_pixels=int(pred.sum()), iou=intersection / union)
    return row


def end_to_end_counts(row, predicted_ng):
    if row.get("gt_status") != "valid":
        return None
    if not predicted_ng:
        return {"intersection": 0, "union": row["gt_pixels"], "gt_pixels": row["gt_pixels"], "predicted_pixels": 0, "iou": 0.0}
    return {k: row[k] for k in ("intersection", "union", "gt_pixels", "predicted_pixels", "iou")}


def aggregate_segmentation(rows, predictions):
    pairs = list(zip(rows, predictions))
    invalid = sum(r.get("gt_status") not in ("valid", "outside_roi", "not_required") for r, _ in pairs)
    counts = [c for r, p in pairs if (c := end_to_end_counts(r, p)) is not None]
    intersection = sum(c["intersection"] for c in counts)
    union = sum(c["union"] for c in counts)
    pixels = sum(c["gt_pixels"] + c["predicted_pixels"] for c in counts)
    valid = bool(counts) and not invalid
    return {"iou_micro": intersection / union if valid else None,
            "dice_micro": 2 * intersection / pixels if valid else None,
            "mean_iou_all_ng": float(np.mean([c["iou"] for c in counts])) if valid else None,
            "segmentation_valid_ng": len(counts), "segmentation_invalid_ng": invalid,
            "segmentation_status": "incomplete_gt_or_prediction" if invalid else "valid" if counts else "no_valid_ng",
            "intersection": intersection, "union": union}
