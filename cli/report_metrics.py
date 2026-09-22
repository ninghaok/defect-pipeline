from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np

from _common import context
from detected_pipeline.util import atomic_write_json
from detected_pipeline.config import roi_mask_for
from detected_pipeline.online_metrics import VERIFIED_SOURCES, reviewed_metrics


def category_metrics(workspace, category, roi_mask=None):
    with sqlite3.connect(workspace / "state" / "pipeline.sqlite3") as db:
        db.row_factory = sqlite3.Row
        records = db.execute("SELECT * FROM samples WHERE category=?", (category,)).fetchall()
    rows, latencies = [], []
    excluded_initialization = excluded_unverified = missing_predictions = 0
    for row in records:
        if row["camera_id"] == "initialization" or row["label_source"] == "initial_calibration_yolo_train_ok":
            excluded_initialization += 1
            continue
        if row["label_source"] not in VERIFIED_SOURCES or row["reviewed_decision"] not in ("OK", "NG"):
            excluded_unverified += 1
            continue
        prediction_path = workspace / "inference_results" / category / f"{row['sample_id']}.json"
        if not prediction_path.is_file():
            missing_predictions += 1
            continue
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        rows.append({"image": row["source_path"], "truth": row["reviewed_decision"], "label_source": row["label_source"],
                     "gt_mask": row["mask_path"], "official": row["model_decision"], "official_mask": prediction.get("binary_mask_path")})
        if prediction.get("latency_ms") is not None:
            latencies.append(float(prediction["latency_ms"]))
    result = reviewed_metrics(rows, roi_mask=roi_mask)
    classification, segmentation = result["classification"], result["segmentation"]
    tn, fp = classification["tn"], classification["fp"]
    return {"category": category, **result["classification"], **result["segmentation"],
            "metrics_schema": 2,
            # Preserve old field meanings for downstream consumers; primary IoU is explicit.
            "false_positive_rate": classification["ok_false_positive_rate"],
            "specificity": tn / (tn + fp) if tn + fp else None,
            "ng_mask_count": segmentation["segmentation_valid_ng"],
            "iou_mean": segmentation["mean_iou_all_ng"],
            "excluded_initialization": excluded_initialization, "excluded_unverified": excluded_unverified,
            "missing_prediction_records": missing_predictions,
            "latency_ms_mean": float(np.mean(latencies)) if latencies else None}


def main() -> None:
    _, workspace, categories, config = context()
    report_root = workspace.parent / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    results = [category_metrics(workspace, category, roi_mask_for(config, category)) for category in categories]
    atomic_write_json(report_root / "classification_metrics.json", results)
    with (report_root / "classification_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]) if results else ["category"])
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
