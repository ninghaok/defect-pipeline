"""Threshold rules shared by the pretrained detector and the YOLO-seg detector.

All rules work on image scores (higher = more anomalous) and are calibrated on
images that never enter training.  Three stages, selected by the number of
confirmed NG available for the category:

* ``ok_quantile``   zero NG      threshold = quantile q of calibration OK scores
* ``ladder``        few NG       OK-quantile ladder; the highest rung that still catches every known NG
* ``recall_first``  enough NG    lowest OK false-positive rate among thresholds reaching the target recall,
                                 under an FPR cap; when the target is unreachable under the cap the Youden
                                 optimum (recall - FPR) under the cap is used instead of the cap edge.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

LADDER_LEVELS = (1.0, 0.95, 0.9, 0.85, 0.8)
DEFAULT_QUANTILES = (0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999)


def quantile_higher(values: Sequence[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot take a quantile of an empty sample")
    return float(np.quantile(array, q, method="higher"))


def classification_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=bool); p = np.asarray(list(predictions), dtype=bool)
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum()); tn = int((~p & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "ok_false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "accuracy": (tp + tn) / len(y) if len(y) else None}


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    s = np.asarray(list(scores), dtype=np.float64); y = np.asarray(list(labels), dtype=bool)
    if y.all() or not y.any():
        return None
    pos, neg = s[y], s[~y]
    greater = (pos[:, None] > neg[None, :]).sum(); equal = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def ok_quantile_threshold(ok_scores: Sequence[float], quantile: float) -> dict[str, Any]:
    threshold = quantile_higher(ok_scores, quantile)
    return {"rule": "ok_quantile", "threshold": threshold, "quantile": float(quantile),
            "calibration_ok": len(list(ok_scores)), "calibration_ng": 0}


def ladder_threshold(ok_scores: Sequence[float], ng_scores: Sequence[float],
                     quantiles: Sequence[float] = DEFAULT_QUANTILES) -> dict[str, Any]:
    """Highest OK quantile whose threshold still catches every known NG; fallback = lowest rung."""
    ng = np.asarray(list(ng_scores), dtype=np.float64)
    if ng.size == 0:
        raise ValueError("ladder rule needs at least one NG score")
    rungs = [(float(q), quantile_higher(ok_scores, q)) for q in sorted(quantiles)]
    fitting = [(q, t) for q, t in rungs if bool((ng >= t).all())]
    q, t = max(fitting, key=lambda x: x[0]) if fitting else rungs[0]
    return {"rule": "ladder", "threshold": t, "quantile": q, "all_ng_caught": bool(fitting),
            "ng_missed_at_fallback": int((ng < t).sum()), "rungs": [{"quantile": a, "threshold": b} for a, b in rungs],
            "calibration_ok": len(list(ok_scores)), "calibration_ng": int(ng.size)}


def recall_first(scores: Sequence[float], labels: Sequence[bool], target_recall: float = 0.95,
                 max_fpr: float = 0.2, min_threshold: float = 0.0) -> dict[str, Any]:
    """Recall-first image threshold under an FPR cap (see module docstring).

    Scores <= ``min_threshold`` are never valid thresholds (for YOLO ``0`` means
    "no detection", so a zero-score NG is a model miss rather than something a
    threshold may hide).
    """
    if not 0.0 < target_recall <= 1.0 or not 0.0 < max_fpr <= 1.0:
        raise ValueError("target_recall and max_fpr must be within (0, 1]")
    s = np.asarray(list(scores), dtype=np.float64); y = np.asarray(list(labels), dtype=bool)
    if not y.any() or y.all():
        raise ValueError("recall_first needs both OK and NG scores")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinity")
    candidates = sorted(set(float(v) for v in s if v > min_threshold))
    rows = []
    for t in candidates + [float(np.nextafter(s.max(), np.inf))]:
        m = classification_metrics(y, s >= t); rows.append({"threshold": t, **m})
    capped = [r for r in rows if r["ok_false_positive_rate"] <= max_fpr]
    eligible = [r for r in capped if r["recall"] >= target_recall]
    if eligible:
        best = min(eligible, key=lambda r: (r["ok_false_positive_rate"], -r["recall"], -r["threshold"]))
        status = "target_recall_achieved"
    else:
        best = max(capped, key=lambda r: (r["recall"] - r["ok_false_positive_rate"], r["recall"], r["threshold"]))
        status = "target_recall_unreachable_under_cap_youden_fallback"
    uncapped = [r for r in rows if r["recall"] >= target_recall]
    uncapped_best = min(uncapped, key=lambda r: (r["ok_false_positive_rate"], -r["threshold"])) if uncapped else None
    ladder = []
    for level in LADDER_LEVELS:
        fitting = [r for r in rows if r["recall"] >= level]
        pick = min(fitting, key=lambda r: (r["ok_false_positive_rate"], -r["threshold"])) if fitting else None
        ladder.append({"recall_level": level, "threshold": pick["threshold"] if pick else None,
                       "ok_false_positive_rate": pick["ok_false_positive_rate"] if pick else None})
    ng = s[y]
    return {"rule": "recall_first", **best, "status": status, "target_recall": float(target_recall),
            "max_fpr": float(max_fpr), "reachable_under_cap": bool(eligible),
            "fpr_cap_binding": bool(uncapped_best is not None and uncapped_best["ok_false_positive_rate"] > max_fpr),
            "auroc": auroc(s, y), "zero_score_ng": int((ng <= min_threshold).sum()),
            "ng_below_threshold": int((ng < best["threshold"]).sum()), "recall_ladder": ladder,
            "calibration_ok": int((~y).sum()), "calibration_ng": int(y.sum())}


def choose_image_threshold(ok_scores: Sequence[float], ng_scores: Sequence[float], rules: dict[str, Any],
                           category: str, min_threshold: float = 0.0) -> dict[str, Any]:
    """Select the rule by NG count: 0 -> OK quantile, < full_rule_min_ng -> ladder, else recall-first."""
    def per_category(key: str, default: float) -> float:
        value = rules.get(key, default)
        if isinstance(value, dict):
            return float(value.get(category, value.get("default", default)))
        return float(value)
    ng = list(ng_scores); ok = list(ok_scores)
    if len(ok) < int(rules.get("min_calibration_ok", 50)):
        raise ValueError(f"{category}: need at least {rules.get('min_calibration_ok', 50)} calibration OK, got {len(ok)}")
    if not ng:
        result = ok_quantile_threshold(ok, per_category("zero_ng_quantile", 0.90))
    elif len(ng) < int(rules.get("full_rule_min_ng", 30)):
        result = ladder_threshold(ok, ng, tuple(rules.get("ladder_quantiles", DEFAULT_QUANTILES)))
    else:
        result = recall_first(ok + ng, [False] * len(ok) + [True] * len(ng), per_category("target_recall", 0.95),
                              float(rules.get("max_fpr", 0.2)), min_threshold)
    result["category"] = category
    return result


def mask_conf_threshold(ious_by_conf: dict[float, list[float]]) -> dict[str, Any]:
    """Confidence threshold for kept instances: the grid value with the best mean NG IoU (ties -> higher)."""
    rows = [{"mask_conf_threshold": float(t), "mean_iou": float(np.mean(v)) if v else 0.0} for t, v in sorted(ious_by_conf.items())]
    best = max(rows, key=lambda r: (r["mean_iou"], r["mask_conf_threshold"]))
    return {**best, "curve": rows}
