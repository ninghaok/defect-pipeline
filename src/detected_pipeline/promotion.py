"""Promotion on paired, reviewed shadow samples; calibration only sets thresholds."""
from detected_pipeline.metric_support import SEGMENTATION_METRIC_VERSION, TOLERANCE_PIXELS
from detected_pipeline.online_metrics import aggregate_reviewed_metrics
from detected_pipeline.util import utc_now

PROMOTION_RULE = "shadow_cost_t5_capped_gain20_v3"
TOLERANCE_GAIN_POINTS = 20


def compare_models(official_rows, candidate_rows):
    """Lower cost, a better equal-cost T5, or a T5 gain of at least 20 points wins.

    Both models must be evaluated on the same verified OK/NG cohort. Test-set
    metrics never enter this gate. Cross-products avoid rounded tie-break gains.
    """
    report = {"rule": PROMOTION_RULE, "decision": "reject", "checks": {}, "compared_at": utc_now()}
    required = {"sample_id", "label", "gt_status", "predicted_ng"}
    if not all(required <= row.keys() for row in [*official_rows, *candidate_rows]):
        return {**report, "reason": "missing_or_legacy_metric_rows"}
    old_ids = [(r["sample_id"], r["label"]) for r in official_rows]
    new_ids = [(r["sample_id"], r["label"]) for r in candidate_rows]
    if old_ids != new_ids or len(set(x[0] for x in old_ids)) != len(old_ids):
        return {**report, "reason": "unpaired_samples"}
    if any(r.get("segmentation_metric_version") != SEGMENTATION_METRIC_VERSION
           or r.get("tolerance_pixels") != TOLERANCE_PIXELS
           for r in [*official_rows, *candidate_rows] if r.get("gt_status") == "valid"):
        return {**report, "reason": "missing_or_legacy_tolerance_counts"}
    metrics = [aggregate_reviewed_metrics(rows) for rows in (official_rows, candidate_rows)]
    old, new = [{**m["classification"], **m["segmentation"], "scope": "paired_promotion_cohort"} for m in metrics]
    report.update(official=old, candidate=new)
    complete = all(m["recall"] is not None and m["ok_false_positive_rate"] is not None
                   and m["tolerant_agreement_micro"] is not None and m["segmentation_status"] == "valid"
                   and m["excluded_invalid_gt"] == 0 for m in (old, new))
    if not complete:
        return {**report, "checks": {"metrics_available": False}, "reason": "insufficient_or_invalid_metrics"}
    # Identical sample/label sequence guarantees equal NG and OK denominators.
    better_tolerance = new["tolerant_matched_pixels"] * old["tolerant_total_pixels"] > old["tolerant_matched_pixels"] * new["tolerant_total_pixels"]
    lower_cost = new["error_cost"] < old["error_cost"]
    equal_cost = new["error_cost"] == old["error_cost"]
    # T_new - T_old >= 20/100, evaluated exactly from pooled pixel counts.
    tolerance_gain = (new["tolerant_matched_pixels"] * old["tolerant_total_pixels"]
                      - old["tolerant_matched_pixels"] * new["tolerant_total_pixels"])
    large_tolerance_gain = (100 * tolerance_gain >= TOLERANCE_GAIN_POINTS
                            * old["tolerant_total_pixels"] * new["tolerant_total_pixels"])
    promote = lower_cost or (equal_cost and better_tolerance) or large_tolerance_gain
    reason = ("lower_error_cost" if lower_cost else "equal_cost_better_tolerance" if equal_cost and better_tolerance
              else "tolerance_gain_at_least_20_points" if large_tolerance_gain else "higher_error_cost" if not equal_cost else "equal_cost_no_tolerance_gain")
    return {**report, "decision": "promote" if promote else "reject", "reason": reason,
            "checks": {"metrics_available": True, "lower_error_cost": lower_cost,
                       "equal_error_cost": equal_cost, "tolerance_improved": better_tolerance,
                       "tolerance_gain_at_least_20_points": large_tolerance_gain},
            "improved_metrics": (["error_cost"] if lower_cost else []) + (["tolerant_agreement_micro"] if better_tolerance else [])}
