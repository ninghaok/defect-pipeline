from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

from _common import context
from detected_pipeline.util import atomic_write_json
from detected_pipeline.masks import external_gt,internal_mask


def ratio(a: int, b: int) -> float:
    return a / b if b else 0.0


def main() -> None:
    _, workspace, categories, _ = context()
    report_root = workspace.parent / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    database = workspace / "state" / "pipeline.sqlite3"
    results = []
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        for category in categories:
            rows = db.execute(
                "SELECT * FROM samples WHERE category=? AND reviewed_decision IS NOT NULL", (category,)
            ).fetchall()
            tp = sum(row["model_decision"] == "NG" and row["reviewed_decision"] == "NG" for row in rows)
            fp = sum(row["model_decision"] == "NG" and row["reviewed_decision"] == "OK" for row in rows)
            fn = sum(row["model_decision"] == "OK" and row["reviewed_decision"] == "NG" for row in rows)
            tn = sum(row["model_decision"] == "OK" and row["reviewed_decision"] == "OK" for row in rows)
            dice_values, iou_values, latencies = [], [], []
            for row in rows:
                prediction_path = workspace / "inference_results" / category / f"{row['sample_id']}.json"
                if not prediction_path.is_file():
                    continue
                prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
                latencies.append(float(prediction.get("latency_ms", 0.0)))
                if row["reviewed_decision"] != "NG" or not row["mask_path"]:
                    continue
                predicted_mask = Path(prediction["binary_mask_path"])
                truth_mask = Path(row["mask_path"])
                if not predicted_mask.is_file() or not truth_mask.is_file():
                    continue
                truth = external_gt(truth_mask)
                predicted = internal_mask(predicted_mask,truth.shape)
                intersection = int(np.logical_and(truth, predicted).sum())
                truth_count, predicted_count = int(truth.sum()), int(predicted.sum())
                union = truth_count + predicted_count - intersection
                dice_values.append(ratio(2 * intersection, truth_count + predicted_count))
                iou_values.append(ratio(intersection, union))
            results.append({
                "category": category, "count": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "recall": ratio(tp, tp + fn), "false_positive_rate": ratio(fp, fp + tn),
                "precision": ratio(tp, tp + fp), "specificity": ratio(tn, tn + fp),
                "accuracy": ratio(tp + tn, len(rows)),
                "ng_mask_count": len(dice_values),
                "dice_mean": float(np.mean(dice_values)) if dice_values else None,
                "iou_mean": float(np.mean(iou_values)) if iou_values else None,
                "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
            })
    atomic_write_json(report_root / "classification_metrics.json", results)
    with (report_root / "classification_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]) if results else ["category"])
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
