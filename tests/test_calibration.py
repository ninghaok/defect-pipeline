import numpy as np
import pytest

from detected_pipeline.calibration import (choose_image_threshold, ladder_threshold, mask_conf_threshold,
                                           ok_quantile_threshold, recall_first)


def test_recall_first_minimizes_fpr_after_target():
    scores = [.20, .50, .80, .10, .30, .70]; labels = [True, True, True, False, False, False]
    result = recall_first(scores, labels, target_recall=2 / 3, max_fpr=1.0)
    assert result["threshold"] == .50 and result["status"] == "target_recall_achieved"
    assert result["recall"] == pytest.approx(2 / 3) and result["ok_false_positive_rate"] == pytest.approx(1 / 3)


def test_recall_first_never_uses_zero_and_counts_zero_score_ng():
    scores = [0.0, .7, 0.0, .2]; labels = [True, True, False, False]
    result = recall_first(scores, labels, target_recall=.95, max_fpr=.2)
    assert result["threshold"] > 0 and result["zero_score_ng"] == 1 and result["recall"] == .5


def test_recall_first_youden_fallback_does_not_run_to_the_cap_edge():
    # 10 NG: 8 well separated, 2 buried inside the OK bulk; cap 0.5 would allow catching them at FPR 0.45
    ng = [.9] * 8 + [.02, .03]; ok = list(np.linspace(.01, .10, 20))
    result = recall_first(ok + ng, [False] * 20 + [True] * 10, target_recall=.95, max_fpr=.5)
    assert result["status"].startswith("target_recall_unreachable")
    assert result["recall"] == .8 and result["ok_false_positive_rate"] == 0.0


def test_ladder_picks_highest_rung_catching_all_ng_and_falls_back():
    ok = list(np.linspace(0, 1, 1001))
    caught = ladder_threshold(ok, [.98, .995], quantiles=(.8, .9, .95, .99))
    assert caught["quantile"] == .95 and caught["all_ng_caught"]
    fallback = ladder_threshold(ok, [.5], quantiles=(.8, .9))
    assert fallback["quantile"] == .8 and not fallback["all_ng_caught"] and fallback["ng_missed_at_fallback"] == 1


def test_choose_image_threshold_switches_rule_by_ng_count():
    ok = list(np.linspace(0, 1, 200)); rules = {"zero_ng_quantile": {"default": .9, "wa_yuan_detection": .95},
                                                 "full_rule_min_ng": 3, "target_recall": .95, "max_fpr": .2}
    assert choose_image_threshold(ok, [], rules, "wa_yuan_detection")["quantile"] == .95
    assert choose_image_threshold(ok, [.99], rules, "qiumian_fupai")["rule"] == "ladder"
    assert choose_image_threshold(ok, [.99, .98, .97], rules, "qiumian_fupai")["rule"] == "recall_first"
    assert ok_quantile_threshold(ok, .9)["threshold"] >= .9


def test_mask_conf_threshold_prefers_best_iou_then_higher_conf():
    best = mask_conf_threshold({.1: [.3, .4], .5: [.35, .35], .9: [.1]})
    assert best["mask_conf_threshold"] == .5 and best["mean_iou"] == pytest.approx(.35)
