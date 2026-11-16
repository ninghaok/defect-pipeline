"""Post-run simulation evaluation; retain the original 15 columns and append audit fields.

Reads full folder ground truth ONLY after lifecycle processing. Never writes feedback,
thresholds, model registry or promotion decisions. JSON preserves raw numerators/counts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from detected_pipeline.config import load_project_config, roi_mask_for
from detected_pipeline.online_metrics import reviewed_metrics
from detected_pipeline.util import atomic_write_json
from detected_pipeline.promotion import PROMOTION_RULE
from detected_pipeline.reporting import model_evaluations
from detected_pipeline.metric_support import SEGMENTATION_METRIC_VERSION
from prepare_simulation_streams import find_mask

HEADERS = ["批次", "输入数", "实际的输入OK/NG数", "模型推理的OK/NG数", "需要人工打标的数量",
           "可用于模型训练OK/NG数", "正式模型", "在线Recall", "在线OK误报率", "在线micro IoU",
           "最新YOLO（状态）", "该YOLO实际训练OK/NG数", "YOLO独立评测Recall", "YOLO独立评测OK误报率", "YOLO独立评测micro IoU"]
HEADERS += ["类别", "实验方案", "在线漏检数", "在线误报数", "独立评测漏检数", "独立评测误报数",
            "该YOLO实际训练真实NG数", "该YOLO实际加入伪缺陷数", "伪缺陷占训练集比例", "累计生成轮数",
            "影子阶段状态", "增强选择说明"]
HEADERS += ["影子漏检（正式→候选）", "影子误报（正式→候选）", "影子micro IoU（正式→候选）", "影子错误成本（正式→候选）", "影子容差吻合率5px（正式→候选）"]
HEADERS += ["在线错误成本", "在线容差吻合率5px", "独立评测错误成本", "独立评测容差吻合率5px"]
STATUS = {"shadow_candidate": "影子中", "promoted": "已晋升", "production": "正式模型", "rejected_shadow": "影子拒绝", "superseded": "被新版本替代"}


def model_name(version):
    if version == "pretrained":
        return "预训练模型"
    return "YOLO-v" + version.split("-seg-v", 1)[1].split("-", 1)[0]


def gate_pairs(gate):
    if gate.get("rule") != PROMOTION_RULE:
        return [None] * 5
    old, new = gate.get("official", {}), gate.get("candidate", {})
    result = []
    for key in ("fn", "fp", "iou_micro", "error_cost", "tolerant_agreement_micro"):
        a, b = old.get(key), new.get(key)
        result.append(None if a is None or b is None else f"{a:.4%} → {b:.4%}" if key in ("iou_micro", "tolerant_agreement_micro") else f"{a} → {b}")
    return result


def export_category(workspace, category, config, output):
    summaries = {}
    for path in (workspace / "model_registry" / category / "milestones").glob("*/summary.json"):
        value = json.loads(path.read_text(encoding="utf-8")); summaries[value["model_version"]] = value
    detailed, table, pending = [], [], []
    for path in sorted((workspace / "batch_reports" / category).glob("batch_*.json")):
        report = json.loads(path.read_text(encoding="utf-8")); snapshot = report.get("end_of_batch")
        if not snapshot or not snapshot.get("lifecycle_complete"):
            pending.append(report["batch"])
            continue
        rows = []
        for item in report["rows"]:
            image = Path(item["image"]); label = image.parent.name.upper()
            if label not in ("OK", "NG"):
                raise ValueError(f"Simulation truth unavailable: {image}")
            gt = find_mask(image.parent.parent / "mask", image) if label == "NG" else None
            rows.append({**item, "truth": label, "gt_mask": str(gt) if gt else None, "label_source": "folder_ground_truth"})
        evaluated = reviewed_metrics(rows, roi_mask=roi_mask_for(config, category))
        for metrics in evaluated.values(): metrics["scope"] = "complete_simulated_batch"
        ng = sum(r["truth"] == "NG" for r in rows); predicted_ng = sum(r["official"] == "NG" for r in rows)
        latest = snapshot["latest_yolo"]; fixed = latest.get("fixed_test") or {} if latest else {}
        train_counts = None
        if latest:
            stats = summaries[latest["model_version"]]["dataset_stats"]["train"]
            train_counts = f"{stats['images'] - stats['ng']}/{stats['ng']}"
        summary = summaries[latest["model_version"]] if latest else {}
        augmentation = summary.get("augmentation", {})
        selected = augmentation.get("selected_count", 0)
        metadata_path = workspace / "model_registry" / category / "versions" / (latest["model_version"] if latest else "none") / "model.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if latest and metadata_path.is_file() else {}
        values = [report["batch"], len(rows), f"{len(rows)-ng}/{ng}", f"{len(rows)-predicted_ng}/{predicted_ng}", report["reviewed"],
                  f"{snapshot['eligible_training_ok']}/{snapshot['eligible_training_ng']}", model_name(report["official_model"]),
                  evaluated["classification"]["recall"], evaluated["classification"]["ok_false_positive_rate"], evaluated["segmentation"]["iou_micro"],
                  f"{model_name(latest['model_version'])}（{STATUS.get(latest['status'], latest['status'])}）" if latest else "—",
                  train_counts, fixed.get("recall"), fixed.get("ok_false_positive_rate"), fixed.get("iou_micro")]
        values += [config.get("categories", {}).get(category, {}).get("display_name", category),
                   "新规则+伪缺陷" if config.get("synthetic", {}).get("enabled") else "未加伪缺陷",
                   evaluated["classification"]["fn"], evaluated["classification"]["fp"], fixed.get("fn"), fixed.get("fp"),
                   stats["ng"] - selected if latest else None, selected if latest else None,
                   selected / stats["images"] if latest and stats["images"] else None,
                   augmentation.get("event_count", 0) if latest else None,
                   STATUS.get(latest["status"], latest["status"]) if latest else "无候选模型",
                   augmentation.get("reason", "未启用") if latest else "尚未训练"]
        # Do not leak a later shadow verdict into earlier batch snapshots.
        shadow = metadata.get("shadow_comparison", {}) if latest and latest["status"] in ("promoted", "production", "rejected_shadow") else {}
        values += gate_pairs(shadow)
        values += [evaluated["classification"]["error_cost"], evaluated["segmentation"]["tolerant_agreement_micro"],
                   fixed.get("error_cost"), fixed.get("tolerant_agreement_micro")
                   if fixed.get("segmentation_metric_version") == SEGMENTATION_METRIC_VERSION else None]
        table.append(dict(zip(HEADERS, values)))
        detailed.append({"batch": report["batch"], "full_stream": evaluated, "reviewed_subset": report["official"],
                         "snapshot": snapshot, "model_usage": report.get("model_usage"),
                         "source_report": str(path), "table_row": table[-1]})
    models = [json.loads(p.read_text(encoding="utf-8"))
              for p in (workspace / "model_registry" / category / "versions").glob("*/model.json")]
    test_reports = [json.loads(p.read_text(encoding="utf-8"))
                    for p in (workspace / "test_reports" / category).glob("*/report.json")]
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / f"{category}_table.json", {"category": category, "segmentation_metric_version": SEGMENTATION_METRIC_VERSION,
                      "headers": HEADERS, "rows": table,
                      "pending_batches": pending, "details": detailed,
                      "independent_evaluations": model_evaluations(test_reports, models)})
    with (output / f"{category}_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS); writer.writeheader(); writer.writerows(table)
    return {"category": category, "completed_rows": len(table), "pending_batches": pending, "output": str(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_category(args.workspace, args.category, load_project_config(PROJECT), args.output), ensure_ascii=False))


if __name__ == "__main__": main()
