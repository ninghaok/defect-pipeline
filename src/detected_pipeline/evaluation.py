"""Fixed test set evaluation: report-only, never used for training, calibration or model selection.

Per-image scores, predicted masks and visuals depend on the model (and its pixel/mask threshold) but not
on the image threshold, so they are produced once per model key and cached under ``test_cache``; changing
the image threshold (pretrained recalibration) re-derives the metrics and the tp/fp/fn/tn folders without
re-running inference.  Report folders hard-link the cached files.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from detected_pipeline.calibration import auroc, classification_metrics
from detected_pipeline.cache_identity import test_identity
from detected_pipeline.metric_support import aggregate_segmentation, end_to_end_counts, segmentation_row
from detected_pipeline.masks import internal_mask
from detected_pipeline.roi import read_image, write_image
from detected_pipeline.util import atomic_write_json, utc_now

# predict(path) -> {"score": float, "mask": bool HxW | None, "heat": float HxW | None, "boxes": [[x1,y1,x2,y2,label]], "extra": {...}}
PredictFn = Callable[[Path], dict[str, Any]]
CASES = ("tp", "fp", "fn", "tn")


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _visuals(image: np.ndarray, heat: np.ndarray | None, mask: np.ndarray | None, boxes: list, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if mask is not None:
        write_image(folder / "pred_mask.png", mask.astype(np.uint8) * 255)
    overlay = image.copy()
    if heat is not None:
        finite = np.isfinite(heat); low, high = float(heat[finite].min()), float(heat[finite].max())
        normalized = np.zeros(heat.shape, np.uint8)
        if high > low:
            normalized = np.round((np.clip(heat, low, high) - low) / (high - low) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        write_image(folder / "heatmap.jpg", colored)
        overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)
    if mask is not None and mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    for x1, y1, x2, y2, label in boxes:
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(overlay, str(label), (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    write_image(folder / "boxed.jpg", overlay)


def score_fixed_test(items: list[dict[str, Any]], predict: PredictFn | None, cache_dir: Path, model_key: str,
                     roi_mask: Path | None = None, visuals: bool = True, progress: Callable[[str], None] | None = None,
                     inference_settings: dict | None = None) -> list[dict[str, Any]]:
    """Run (or load from cache) the model on every test item.  Items: {image, label ('OK'|'NG'), mask}."""
    cache_path = cache_dir / "scores.json"
    identity = test_identity(items, model_key, roi_mask, inference_settings)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("identity") == identity and (not visuals or cached.get("visuals")):
            return cached["rows"]
    if predict is None:
        raise RuntimeError(f"no valid fixed-test cache for {model_key} and no predictor given")
    rows = []
    for index, item in enumerate(items):
        path = Path(item["image"]); result = predict(path); mask = result.get("mask")
        image = read_image(path)
        if mask is not None and roi_mask:
            mask = mask & internal_mask(roi_mask, image.shape[:2])
        row = {"index": index, "image": str(path), "label": item["label"], "score": float(result["score"]), "iou": None,
               "gt_mask": item.get("mask"), "extra": result.get("extra", {})}
        row.update(segmentation_row(item["label"], item.get("mask"), mask, image.shape[:2], roi_mask))
        if visuals:
            # A new inference identity must not overwrite hard-linked historical visuals.
            folder = cache_dir / "visuals" / identity / f"{index:05d}"
            row["visuals_dir"] = str(folder)
            _visuals(image, result.get("heat"), mask, result.get("boxes", []), folder)
        rows.append(row)
        if progress and (index + 1) % 50 == 0:
            progress(f"FIXED TEST {index + 1}/{len(items)}")
    atomic_write_json(cache_path, {"model_key": model_key, "identity": identity, "scored_at": utc_now(), "visuals": visuals, "rows": rows})
    return rows


def case_of(row: dict[str, Any], image_threshold: float) -> str | None:
    if row["label"] not in ("OK", "NG"):
        return None
    predicted = row["score"] >= image_threshold; actual = row["label"] == "NG"
    return "tp" if actual and predicted else "fn" if actual else "fp" if predicted else "tn"


def fixed_test_metrics(rows: list[dict[str, Any]], image_threshold: float) -> dict[str, Any]:
    used = [r for r in rows if r["label"] in ("OK", "NG")]
    labels = [r["label"] == "NG" for r in used]; scores = [r["score"] for r in used]
    predicted = [s >= image_threshold for s in scores]
    result = classification_metrics(labels, predicted)
    segmentation = aggregate_segmentation(rows, [r["score"] >= image_threshold for r in rows])
    result.update({"test_auroc": auroc(scores, labels), "image_threshold": float(image_threshold),
                   **segmentation, "primary_segmentation_metric": "iou_micro", "metrics_schema": 2,
                   "test_ok": int(sum(not l for l in labels)), "test_ng": int(sum(labels)),
                   "excluded_invalid_gt": sum(r["label"] == "INVALID_GT" for r in rows),
                   "excluded_outside_roi": int(sum(r["label"] == "EXCLUDED" for r in rows))})
    return result


def write_test_report(workspace: Path, category: str, model_version: str, role: str, rows: list[dict[str, Any]],
                      image_threshold: float, cache_dir: Path | None = None, extra: dict[str, Any] | None = None) -> Path:
    """Metrics json + tp/fp/fn/tn folders (original, original mask, cached pred mask / heatmap / boxed, score.json)."""
    metrics = fixed_test_metrics(rows, image_threshold)
    report = {"category": category, "model_version": model_version, "role": role, "at": utc_now(), **metrics, **(extra or {})}
    stamp = report["at"].replace(":", "").replace("-", "").replace(".", "")
    folder = workspace / "test_reports" / category / f"{stamp}_{role}_{model_version}"
    folder.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in rows:
        case = case_of(row, image_threshold)
        if case is None:
            continue
        image = Path(row["image"]); target = folder / case / f"{row['index']:05d}"
        _link(image, target / f"original{image.suffix.lower()}")
        if row.get("gt_mask") and Path(row["gt_mask"]).is_file():
            gt = Path(row["gt_mask"]); _link(gt, target / f"original_mask{gt.suffix.lower()}")
        if cache_dir is not None:
            source = Path(row["visuals_dir"]) if row.get("visuals_dir") else cache_dir / "visuals" / f"{row['index']:05d}"
            for name in ("pred_mask.png", "heatmap.jpg", "boxed.jpg"):
                if (source / name).exists():
                    _link(source / name, target / name)
        counts = end_to_end_counts(row, row["score"] >= image_threshold)
        details = {"gt_status": row["gt_status"], "raw_localization_iou": row["iou"],
                   "end_to_end_iou": counts["iou"] if counts else None,
                   "intersection": counts["intersection"] if counts else None,
                   "union": counts["union"] if counts else None}
        atomic_write_json(target / "score.json", {**row.get("extra", {}), "image": row["image"], "label": row["label"], "case": case, "score": row["score"],
                                                  "image_threshold": float(image_threshold), "iou": row["iou"], **details})
        cases.append({"case": case, "index": row["index"], "image": row["image"], "label": row["label"], "score": row["score"], "iou": row["iou"], **details})
    with (folder / "cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "index", "image", "label", "score", "iou", "gt_status", "raw_localization_iou", "end_to_end_iou", "intersection", "union"]); writer.writeheader(); writer.writerows(cases)
    report["counts_by_case"] = {c: sum(r["case"] == c for r in cases) for c in CASES}; report["folder"] = str(folder)
    report["gt_issues"] = [{"image": r["image"], "gt_mask": r.get("gt_mask"), "status": r["gt_status"]}
                           for r in rows if r["gt_status"] not in ("valid", "not_required")]
    atomic_write_json(folder / "report.json", report)
    atomic_write_json(workspace / "test_reports" / category / f"{stamp}_{role}_{model_version}.json", report)
    with (workspace / "test_reports" / category / "test_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    return folder
