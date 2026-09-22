"""Stand-alone YOLO-seg training with exactly the lifecycle's method (no feedback store needed).

Data layout (same as dataset_523):
    <dataset-root>/<source_dir>/{train,val,test}/{OK,NG}/*.jpg|bmp|png
    <dataset-root>/<source_dir>/mask/<stem>[_t|_mask].<ext>      black = defect, white = background

Roles (fixed, no leakage): train -> training (OK capped, NG capped); val -> calibration (image threshold,
mask confidence threshold; also Ultralytics validation set without model selection); test -> report only.
Method: yolo26s-seg, fixed epochs (configs/training.yaml), copy_paste 0.3, mosaic 1.0, last.pt; image score =
max instance confidence; image threshold = recall-first under the FPR cap (target per category); mask threshold
= best mean NG IoU.  ROI categories: outside filled white, GT ANDed with the ROI, NG entirely outside excluded.

Example:
    python cli/train_yolo.py --category di_mian_detection --output C:\\ninghao\\results\\train_dimian
    python cli/train_yolo.py --category wa_yuan_detection --train-ng-limit 60 --epochs 100 --skip-test
The resulting summary.json carries checkpoint + thresholds in the format the lifecycle registry uses, so
the model can be used by plugins.yolo_supervised.YoloFeedbackAdapter directly.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from detected_pipeline.evaluation import fixed_test_metrics
from detected_pipeline.metric_support import segmentation_row
from detected_pipeline.config import load_project_config, roi_mask_for
from detected_pipeline.masks import write_internal_from_external
from detected_pipeline.roi import read_image, write_image
from detected_pipeline.training.runner import run_yolo_seg_training, smoke_test_seg
from detected_pipeline.training.seg_lifecycle import calibrate_seg, materialize
from detected_pipeline.util import atomic_write_json, sha256_file, utc_now

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXT) if folder.is_dir() else []


def find_mask(mask_root: Path, image: Path) -> Path:
    candidates = [p for suffix in ("_t", "_mask", "") for ext in sorted(EXT) for p in [mask_root / f"{image.stem}{suffix}{ext}"] if p.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one mask for {image}, found {candidates}")
    return candidates[0]


def source_dir(config: dict, category: str, override: str | None) -> str:
    if override:
        return override
    names = config["folder_ground_truth"]["category_directory_names"].get(category, [category])
    return names[-1]


def internal_masks(ng: list[Path], mask_root: Path, out_dir: Path) -> list[tuple[Path, Path]]:
    """Convert dataset masks (black defect) to the internal convention (white defect) used by materialize."""
    out_dir.mkdir(parents=True, exist_ok=True); items = []
    for image in ng:
        target = out_dir / f"{sha256_file(image)[:16]}.png"
        if not target.exists():
            write_internal_from_external(find_mask(mask_root, image), target)
        items.append((image, target))
    return items


def evaluate(detector, items: list[tuple[Path, bool]], thresholds: dict, out: Path, roi_mask: Path | None, mask_root: Path) -> dict:
    """Report-only test pass; saves boxed/heatmap/mask per image under tp/fp/fn/tn."""
    t, mt = thresholds["image_threshold"], thresholds["mask_conf_threshold"]; rows = []
    for case in ("tp", "fp", "fn", "tn"):
        (out / case).mkdir(parents=True, exist_ok=True)
    for index, (image_path, is_ng) in enumerate(items):
        image = read_image(image_path); score, confs, boxes, masks = detector.infer(image)
        predicted = score >= t; keep = (confs >= mt) if predicted else np.zeros(len(confs), bool)
        mask = detector.union(masks, keep, image.shape[:2])
        metric_row = segmentation_row("NG" if is_ng else "OK", find_mask(mask_root, image_path) if is_ng else None,
                                      mask, image.shape[:2], roi_mask)
        if metric_row["label"] in ("EXCLUDED", "INVALID_GT"):
            rows.append({"image": str(image_path), "score": score, **metric_row})
            continue
        case = "tp" if is_ng and predicted else "fn" if is_ng else "fp" if predicted else "tn"
        folder = out / case / f"{index:05d}"; folder.mkdir(parents=True, exist_ok=True)
        boxed = image.copy()
        for (x1, y1, x2, y2), c in zip(boxes[keep], confs[keep]):
            cv2.rectangle(boxed, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(boxed, f"{c:.2f}", (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        write_image(folder / "boxed.jpg", boxed); write_image(folder / "pred_mask.png", mask.astype(np.uint8) * 255)
        try:
            (folder / ("original" + image_path.suffix)).hardlink_to(image_path)
        except OSError:
            shutil.copy2(image_path, folder / ("original" + image_path.suffix))
        rows.append({"image": str(image_path), "label": "NG" if is_ng else "OK", "prediction": "NG" if predicted else "OK", "case": case,
                     "score": score, "n_instances": int(len(confs)), "n_kept": int(keep.sum()), **metric_row})
        if (index + 1) % 20 == 0:
            print(f"TEST {index + 1}/{len(items)}", flush=True)
    result = fixed_test_metrics(rows, t)
    result.update(mask_conf_threshold=mt, rows=rows)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="train one YOLO-seg model with the lifecycle method")
    p.add_argument("--category", required=True, help="pipeline category name, e.g. di_mian_detection (selects ROI and target recall)")
    p.add_argument("--dataset-root", type=Path, default=Path(r"D:\dataset_523\dataset_523"))
    p.add_argument("--source-dir", default=None, help="dataset folder name under dataset-root (default from configs/pipeline.yaml)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-ok-limit", type=int, default=None, help="default lifecycle.train_ok_limit (400)")
    p.add_argument("--train-ng-limit", type=int, default=0, help="0 = all train NG; otherwise the first N after a seeded shuffle")
    p.add_argument("--epochs", type=int, default=None); p.add_argument("--batch", type=int, default=None); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--skip-test", action="store_true"); p.add_argument("--keep-dataset", action="store_true")
    a = p.parse_args()
    config = load_project_config(PROJECT); life = config["lifecycle"]; training = dict(config["training"])
    if a.category not in config["categories"]:
        p.error(f"unknown category {a.category}; choose from {list(config['categories'])}")
    for key in ("epochs", "batch", "seed"):
        if getattr(a, key) is not None:
            training[key] = getattr(a, key)
    training["patience"] = 0
    seed = int(training["seed"]); rng = random.Random(f"{seed}:{a.category}:train_yolo")
    roi_mask = roi_mask_for(config, a.category)
    root = a.dataset_root / source_dir(config, a.category, a.source_dir); mask_root = root / "mask"
    if not root.is_dir():
        raise FileNotFoundError(root)
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"already trained: {out / 'summary.json'}"); return

    train_ok, train_ng = images(root / "train" / "OK"), images(root / "train" / "NG")
    cal_ok, cal_ng = images(root / "val" / "OK"), images(root / "val" / "NG")
    test_ok, test_ng = images(root / "test" / "OK"), images(root / "test" / "NG")
    rng.shuffle(train_ok); rng.shuffle(train_ng)
    train_ok = train_ok[:int(a.train_ok_limit or life.get("train_ok_limit", 400))]
    if a.train_ng_limit:
        train_ng = train_ng[:a.train_ng_limit]
    if not train_ng or not cal_ng or not cal_ok:
        raise ValueError("need train NG, val OK and val NG")
    masks_dir = out / "internal_masks"
    train_items = internal_masks(train_ng, mask_root, masks_dir) + [(p, None) for p in train_ok]
    cal_items = [(p, None) for p in cal_ok] + internal_masks(cal_ng, mask_root, masks_dir)
    dataset_root = out / "dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    yaml_path, stats = materialize(dataset_root, train_items, cal_items, roi_mask)
    print(f"dataset: train {stats['train']} val {stats['val']} excluded {len(stats['excluded'])}", flush=True)
    atomic_write_json(out / "split.json", {"train_ok": [str(p) for p in train_ok], "train_ng": [str(p) for p in train_ng],
                                           "calibration_ok": [str(p) for p in cal_ok], "calibration_ng": [str(p) for p in cal_ng],
                                           "test_ok": [str(p) for p in test_ok], "test_ng": [str(p) for p in test_ng],
                                           "excluded": stats["excluded"], "roi_mask": str(roi_mask) if roi_mask else None})

    started = time.perf_counter()
    checkpoint = run_yolo_seg_training(yaml_path, out / "model", training)
    training_seconds = time.perf_counter() - started
    print(f"trained in {training_seconds / 60:.1f} min: {checkpoint}", flush=True)

    from detected_pipeline.plugins.yolo_supervised import YoloSegDetector
    detector = YoloSegDetector(checkpoint, training.get("inference", {}), roi_mask)
    excluded = {Path(e["image"]) for e in stats["excluded"]}
    calibration = calibrate_seg(detector, [it for it in cal_items if it[0] not in excluded], life.get("yolo_thresholds", {}), a.category, roi_mask)
    thresholds = {"image_threshold": calibration["image_threshold"], "mask_conf_threshold": calibration["mask_conf_threshold"]}
    print(f"thresholds: {thresholds} calibration recall {calibration['classification']['recall']:.3f} "
          f"fpr {calibration['classification']['ok_false_positive_rate']:.3f} auroc {calibration['classification']['auroc']:.3f}", flush=True)
    atomic_write_json(out / "calibration.json", calibration)

    test = None
    if not a.skip_test and (test_ok or test_ng):
        test_items = [(p, False) for p in test_ok] + [(p, True) for p in test_ng if p not in excluded]
        test = evaluate(detector, test_items, thresholds, out / "test", roi_mask, mask_root)
        atomic_write_json(out / "test_report.json", test)
        print(f"test: recall {test['recall']} fpr {test['ok_false_positive_rate']} auroc {test['test_auroc']} "
              f"micro IoU {test['iou_micro']}", flush=True)
    summary = {"category": a.category, "model_version": f"{a.category}-seg-standalone-{utc_now()[:10]}-{sha256_file(checkpoint)[:8]}",
               "checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint), "thresholds": thresholds,
               "smoke_test": smoke_test_seg(checkpoint, cal_items[0][0], training.get("inference", {}), roi_mask),
               "counts": {"train_ok": len(train_ok), "train_ng": len(train_ng), "calibration_ok": len(cal_ok), "calibration_ng": len(cal_ng),
                          "test_ok": len(test_ok), "test_ng": len(test_ng), "excluded_outside_roi": len(stats["excluded"])},
               "training": {k: v for k, v in training.items() if k != "inference"}, "inference": training.get("inference", {}),
               "training_seconds": training_seconds, "roi_mask": str(roi_mask) if roi_mask else None,
               "calibration": {k: v for k, v in calibration.items() if k != "records"},
               "test": {k: v for k, v in (test or {}).items() if k != "rows"} or None, "created_at": utc_now()}
    atomic_write_json(out / "summary.json", summary)
    if not a.keep_dataset:
        shutil.rmtree(dataset_root, ignore_errors=True)
    print(f"DONE: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
