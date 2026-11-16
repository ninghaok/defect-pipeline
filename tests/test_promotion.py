import copy
import json
from pathlib import Path

import numpy as np
import pytest

from detected_pipeline.online_metrics import reviewed_metric_rows
from detected_pipeline.promotion import compare_models
from detected_pipeline.roi import write_image
from detected_pipeline.metric_support import SEGMENTATION_METRIC_VERSION, TOLERANCE_PIXELS


def ng(name, intersection=50, union=100, gt_pixels=100, detected=True):
    return dict(sample_id=name, label="NG", gt_status="valid", predicted_ng=detected,
                intersection=intersection, union=union, gt_pixels=gt_pixels,
                predicted_pixels=union + intersection - gt_pixels, iou=intersection / union,
                tolerant_matched_pixels=intersection,
                tolerant_total_pixels=gt_pixels+min(union-gt_pixels,gt_pixels),
                tolerant_agreement=intersection/(gt_pixels+min(union-gt_pixels,gt_pixels)),
                tolerance_pixels=TOLERANCE_PIXELS, segmentation_metric_version=SEGMENTATION_METRIC_VERSION)


def ok(name, alarm=False):
    return dict(sample_id=name, label="OK", gt_status="not_required", predicted_ng=alarm)


@pytest.mark.parametrize("change,expected,improved", [
    ("iou_only", "promote", ["tolerant_agreement_micro"]),
    ("recall_only", "promote", ["error_cost"]),
    ("small_fpr_gain", "promote", ["error_cost"]),
    ("equal", "reject", []),
    ("worse_iou", "promote", ["error_cost"]),
    ("worse_fp", "reject", ["tolerant_agreement_micro"]),
    ("worse_fn", "reject", []),
])
def test_cost_then_tolerance_gate(change, expected, improved):
    old = [ng("a"), ng("b", intersection=0, detected=False)] + [ok(str(i), i == 0) for i in range(200)]
    new = copy.deepcopy(old)
    if change in ("iou_only", "worse_fp"):
        new[0] = ng("a", intersection=60)
    if change == "recall_only":
        new[1]["predicted_ng"] = True  # detected with an empty mask: classification alone improves
    if change in ("small_fpr_gain", "worse_iou", "worse_fn"):
        new[2]["predicted_ng"] = False
    if change == "worse_iou": new[0] = ng("a", intersection=49)
    if change == "worse_fp": new[3]["predicted_ng"] = True
    if change == "worse_fn": new[0]["predicted_ng"] = False
    result = compare_models(old, new)
    assert result["decision"] == expected
    assert result["improved_metrics"] == improved
    if change == "small_fpr_gain":
        assert result["official"]["ok_false_positive_rate"] == .005
        assert result["candidate"]["ok_false_positive_rate"] == 0


@pytest.mark.parametrize("rows", [[], [ok("ok")], [ng("ng")],
                                    [ng("ng"), {**ok("ok"), "gt_status": "missing_prediction"}],
                                    [{"truth": "NG", "official": "NG", "shadow": "NG"}]])
def test_missing_metrics_and_legacy_state_cannot_promote(rows):
    assert compare_models(rows, rows)["decision"] == "reject"


def test_requires_same_unique_cohort():
    rows = [ng("a"), ok("b")]
    assert compare_models(rows, [ng("c"), ok("b")])["reason"] == "unpaired_samples"
    assert compare_models(rows * 2, rows * 2)["reason"] == "unpaired_samples"


def test_micro_counts_survive_resume_and_are_not_mean_batch_iou():
    old = [ng("small", 1, 1, 1), ng("large", 50, 100, 100), ok("ok")]
    new = [ng("small", 0, 1, 1), ng("large", 60, 100, 100), ok("ok")]
    result = compare_models(json.loads(json.dumps(old)), json.loads(json.dumps(new)))
    assert result["decision"] == "promote"  # macro IoU decreases, requested micro IoU increases
    assert result["official"]["iou_micro"] == pytest.approx(51 / 101)
    assert result["candidate"]["iou_micro"] == pytest.approx(60 / 101)
    assert result["candidate"]["mean_iou_all_ng"] < result["official"]["mean_iou_all_ng"]


def test_exact_iou_ratios_do_not_create_rounding_gains():
    old = [ng("a", 1, 3, 3), ok("ok")]
    new = [ng("a", 2, 6, 3), ok("ok")]
    assert compare_models(old, new)["decision"] == "reject"


def test_shadow_roi_miss_and_missing_mask_rules(tmp_path):
    roi = np.zeros((4, 4), bool); roi[:2, :2] = True
    gt = roi.copy(); gt[3, 3] = True
    image = tmp_path / "image.png"; mask = tmp_path / "gt.png"; rp = tmp_path / "roi.png"
    write_image(image, np.zeros((4, 4, 3), np.uint8))
    write_image(mask, gt.astype(np.uint8) * 255)
    write_image(rp, roi.astype(np.uint8) * 255)
    pred = tmp_path / "pred.png"; write_image(pred, np.ones((4, 4), np.uint8) * 255)
    record = dict(sample_id=str(image), image=str(image), truth="NG", gt_mask=str(mask),
                  official="OK", official_mask=str(pred), label_source="human_review")
    rows = reviewed_metric_rows([record], roi_mask=rp)
    result = compare_models(rows + [ok("normal")], rows + [ok("normal")])
    assert result["official"]["fn"] == 1
    assert result["official"]["iou_micro"] == 0
    assert result["official"]["union"] == 4
    outside = np.zeros((4, 4), np.uint8); outside[3, 3] = 255; write_image(mask, outside)
    record["official"] = "NG"
    assert reviewed_metric_rows([record], roi_mask=rp)[0]["label"] == "EXCLUDED"
    write_image(mask, gt.astype(np.uint8) * 255)
    pred.unlink()
    rows = reviewed_metric_rows([record], roi_mask=rp)
    assert compare_models(rows + [ok("ok")], rows + [ok("ok")])["reason"] == "insufficient_or_invalid_metrics"
    mask.unlink()
    rows = reviewed_metric_rows([record], roi_mask=rp)
    assert compare_models(rows + [ok("ok")], rows + [ok("ok")])["decision"] == "reject"


def test_shadow_does_not_use_pseudo_or_hidden_truth(tmp_path):
    rows = [dict(image="does-not-exist", truth="NG", hidden_truth="NG", official="NG",
                 label_source="sampling_pseudo_ok")]
    assert reviewed_metric_rows(rows) == []


@pytest.mark.parametrize("matched,expected", [(69, "reject"), (70, "promote"), (71, "promote")])
def test_twenty_percentage_point_boundary_with_higher_cost(matched, expected):
    old = [ng("a", 50), ok("b")]
    new = [ng("a", matched), ok("b", alarm=True)]
    result = compare_models(old, new)
    assert result["decision"] == expected
    assert result["checks"]["tolerance_gain_at_least_20_points"] == (expected == "promote")
    assert result["candidate"]["error_cost"] > result["official"]["error_cost"]
    if expected == "promote":
        assert result["reason"] == "tolerance_gain_at_least_20_points"


def test_large_tolerance_gain_can_offset_an_additional_miss():
    old = [ng("a", 10), ng("b", 10), ok("c")]
    new = [ng("a", 100), ng("b", 0, detected=False), ok("c")]
    result = compare_models(old, new)
    assert result["candidate"]["fn"] == 1
    assert result["decision"] == "promote"
    assert result["reason"] == "tolerance_gain_at_least_20_points"


def test_large_gain_still_requires_paired_valid_ok_and_ng():
    assert compare_models([ng("a", 10)], [ng("a", 100)])["decision"] == "reject"
    assert compare_models([ng("a", 10), ok("b")], [ng("other", 100), ok("b")])["decision"] == "reject"
