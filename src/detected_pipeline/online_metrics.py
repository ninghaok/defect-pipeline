"""Metrics for verified online records, separate from feedback/training state."""
from pathlib import Path

from detected_pipeline.calibration import classification_metrics
from detected_pipeline.masks import internal_mask
from detected_pipeline.metric_support import METRICS_SCHEMA, aggregate_segmentation, segmentation_row
from detected_pipeline.roi import read_image

VERIFIED_SOURCES = {"human_review", "folder_ground_truth"}


def reviewed_metric_rows(rows, decision_key="official", mask_key="official_mask", roi_mask=None):
    """Freeze verified per-image counts for reporting and cross-batch promotion."""
    selected = [r for r in rows if r.get("truth") in ("OK", "NG")
                and r.get("label_source") in VERIFIED_SOURCES and r.get(decision_key) in ("OK", "NG")]
    evaluated = []
    for row in selected:
        predicted_ng = row[decision_key] == "NG"
        shape = read_image(Path(row["image"])).shape[:2]
        pred_path = row.get(mask_key)
        available = bool(pred_path and Path(pred_path).is_file())
        pred = internal_mask(pred_path) if available and predicted_ng else None
        result = segmentation_row(row["truth"], row.get("gt_mask"), pred, shape, roi_mask,
                                  external=row["label_source"] == "folder_ground_truth")
        if predicted_ng and result["gt_status"] == "valid" and not available:
            result["gt_status"] = "missing_prediction"
        result["predicted_ng"] = predicted_ng
        result["sample_id"] = row.get("sample_id", row["image"])
        evaluated.append(result)
    return evaluated


def aggregate_reviewed_metrics(evaluated):
    used = [r for r in evaluated if r["label"] in ("OK", "NG")]
    classification = classification_metrics([r["label"] == "NG" for r in used], [r["predicted_ng"] for r in used])
    classification.update(scope="reviewed_online_subset", metrics_schema=METRICS_SCHEMA, count=len(used), verified_count=len(evaluated),
                          excluded_outside_roi=sum(r["label"] == "EXCLUDED" for r in evaluated),
                          excluded_invalid_gt=sum(r["label"] == "INVALID_GT" for r in evaluated))
    segmentation = aggregate_segmentation(evaluated, [r["predicted_ng"] for r in evaluated])
    segmentation.update(scope="reviewed_online_subset", primary_segmentation_metric="tolerant_agreement_micro",
                        images=segmentation["segmentation_valid_ng"], iou=segmentation["iou_micro"], dice=segmentation["dice_micro"])
    return {"classification": classification, "segmentation": segmentation}


def reviewed_metrics(rows, decision_key="official", mask_key="official_mask", roi_mask=None):
    return aggregate_reviewed_metrics(reviewed_metric_rows(rows, decision_key, mask_key, roi_mask))
