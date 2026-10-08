"""Paired calibration/shadow promotion; fixed-test results never enter this module."""
from pathlib import Path

from detected_pipeline.metric_support import segmentation_row
from detected_pipeline.online_metrics import aggregate_reviewed_metrics
from detected_pipeline.roi import read_image
from detected_pipeline.util import utc_now

PROMOTION_RULE = "pareto_recall_fpr_micro_iou_v1"


def score_calibration(items, predict, image_threshold, roi_mask=None):
    """Score (image, internal GT mask or None) with each model's deployed thresholds."""
    rows = []
    for image, gt_mask in items:
        image = Path(image)
        result = predict(image)
        predicted_ng = float(result["score"]) >= image_threshold
        mask = result.get("mask")
        row = segmentation_row("NG" if gt_mask else "OK", gt_mask, mask if predicted_ng else None,
                               read_image(image).shape[:2], roi_mask, external=False)
        if predicted_ng and row["gt_status"] == "valid" and mask is None:
            row["gt_status"] = "missing_prediction"
        row.update(sample_id=str(image), predicted_ng=predicted_ng)
        rows.append(row)
    return rows


def compare_models(official_rows, candidate_rows):
    """Require all three metrics non-worse and at least one strictly better on paired rows.

    Count comparisons and cross-multiplied IoU avoid rounding-dependent minimum gains.
    Missing metrics (including legacy shadow state) never silently become zero.
    """
    report = {"rule": PROMOTION_RULE, "decision": "reject", "checks": {}, "compared_at": utc_now()}
    required = {"sample_id", "label", "gt_status", "predicted_ng"}
    if not all(required <= row.keys() for row in [*official_rows, *candidate_rows]):
        return {**report, "reason": "missing_or_legacy_metric_rows"}
    old_ids = [(r["sample_id"], r["label"]) for r in official_rows]
    new_ids = [(r["sample_id"], r["label"]) for r in candidate_rows]
    if old_ids != new_ids or len(set(x[0] for x in old_ids)) != len(old_ids):
        return {**report, "reason": "unpaired_samples"}
    metrics = [aggregate_reviewed_metrics(rows) for rows in (official_rows, candidate_rows)]
    old, new = [{**m["classification"], **m["segmentation"], "scope": "paired_promotion_cohort"} for m in metrics]
    report.update(official=old, candidate=new)
    complete = all(m["recall"] is not None and m["ok_false_positive_rate"] is not None
                   and m["iou_micro"] is not None and m["segmentation_status"] == "valid"
                   and m["excluded_invalid_gt"] == 0 for m in (old, new))
    if not complete:
        return {**report, "checks": {"metrics_available": False}, "reason": "insufficient_or_invalid_metrics"}
    # Identical sample/label sequence guarantees equal NG and OK denominators.
    old_iou_product = old["intersection"] * new["union"]
    new_iou_product = new["intersection"] * old["union"]
    improved = {"recall": new["tp"] > old["tp"], "ok_false_positive_rate": new["fp"] < old["fp"],
                "iou_micro": new_iou_product > old_iou_product}
    checks = {"metrics_available": True, "no_extra_misses": new["fn"] <= old["fn"],
              "no_extra_false_positives": new["fp"] <= old["fp"],
              "recall_not_worse": new["tp"] >= old["tp"],
              "fpr_not_worse": new["fp"] <= old["fp"],
              "iou_not_worse": new_iou_product >= old_iou_product, "real_gain": any(improved.values())}
    return {**report, "decision": "promote" if all(checks.values()) else "reject", "checks": checks,
            "reason": "pareto_improvement" if all(checks.values()) else "regression_or_no_gain",
            "improved_metrics": [name for name, better in improved.items() if better]}
