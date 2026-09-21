"""Fixed test set evaluation: report-only, never used for training, calibration or model selection.

Per-image scores and NG masks depend on the model (and its pixel/mask threshold) but not on the image
threshold, so they are cached per model key; changing the image threshold (pretrained recalibration)
re-derives the metrics without re-running inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from detected_pipeline.calibration import auroc, classification_metrics
from detected_pipeline.masks import external_gt
from detected_pipeline.roi import read_image
from detected_pipeline.util import atomic_write_json, utc_now

PredictFn = Callable[[Path], tuple[float, np.ndarray | None]]   # (image score, NG mask or None)


def _roi(roi_mask: Path | None, shape, cache: dict):
    if roi_mask is None:
        return None
    if shape not in cache:
        import cv2
        raw = read_image(roi_mask, cv2.IMREAD_GRAYSCALE)
        if raw.shape != shape:
            raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        cache[shape] = raw >= 128
    return cache[shape]


def score_fixed_test(items: list[dict[str, Any]], predict: PredictFn | None, cache_path: Path, model_key: str,
                     roi_mask: Path | None = None, progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Run (or load from cache) the model on every test item.  Items: {image, label ('OK'|'NG'), mask}."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("model_key") == model_key and len(cached.get("rows", [])) == len(items):
            return cached["rows"]
    if predict is None:
        raise RuntimeError(f"no valid fixed-test cache for {model_key} and no predictor given")
    rows = []; roi_cache: dict = {}
    for index, item in enumerate(items):
        path = Path(item["image"]); score, mask = predict(path)
        row = {"image": str(path), "label": item["label"], "score": float(score), "iou": None}
        if item["label"] == "NG" and item.get("mask"):
            gt = external_gt(item["mask"], mask.shape if mask is not None else None)
            roi = _roi(roi_mask, gt.shape, roi_cache)
            if roi is not None:
                gt &= roi
            if not gt.any():
                row["label"] = "EXCLUDED"   # defect entirely outside the ROI: not inspectable by this station
            elif mask is None:
                row["iou"] = 0.0
            else:
                union = int((mask | gt).sum()); row["iou"] = float((mask & gt).sum() / union) if union else 0.0
        rows.append(row)
        if progress and (index + 1) % 50 == 0:
            progress(f"FIXED TEST {index + 1}/{len(items)}")
    atomic_write_json(cache_path, {"model_key": model_key, "scored_at": utc_now(), "rows": rows})
    return rows


def fixed_test_metrics(rows: list[dict[str, Any]], image_threshold: float) -> dict[str, Any]:
    used = [r for r in rows if r["label"] in ("OK", "NG")]
    labels = [r["label"] == "NG" for r in used]; scores = [r["score"] for r in used]
    predicted = [s >= image_threshold for s in scores]
    result = classification_metrics(labels, predicted)
    ng_iou = [(r["iou"] or 0.0) if p else 0.0 for r, p in zip(used, predicted) if r["label"] == "NG"]
    result.update({"test_auroc": auroc(scores, labels), "image_threshold": float(image_threshold),
                   "mean_iou_all_ng": float(np.mean(ng_iou)) if ng_iou else None,
                   "test_ok": int(sum(not l for l in labels)), "test_ng": int(sum(labels)),
                   "excluded_outside_roi": int(sum(r["label"] == "EXCLUDED" for r in rows))})
    return result


def write_test_report(workspace: Path, category: str, model_version: str, role: str, metrics: dict[str, Any],
                      extra: dict[str, Any] | None = None) -> Path:
    report = {"category": category, "model_version": model_version, "role": role, "at": utc_now(), **metrics, **(extra or {})}
    folder = workspace / "test_reports" / category; folder.mkdir(parents=True, exist_ok=True)
    stamp = report["at"].replace(":", "").replace("-", "")[:15]
    path = folder / f"{stamp}_{role}_{model_version}.json"
    atomic_write_json(path, report)
    with (folder / "test_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({k: v for k, v in report.items()}, ensure_ascii=False) + "\n")
    return path
