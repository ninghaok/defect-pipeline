"""YOLO instance-segmentation training (fixed epochs, last.pt) and checkpoint smoke test."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from detected_pipeline.util import atomic_write_json, sha256_file, utc_now


def _validate_segment_dataset(dataset_yaml: Path) -> dict:
    """Fail early if a generated label set is malformed: every image needs a label txt (empty for OK),
    each polygon line is ``class x1 y1 ... xn yn`` with >= 3 normalized points."""
    import yaml
    spec = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(spec.get("path", dataset_yaml.parent))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    nc = len(spec["names"]); images = positives = polygons = 0
    for split in ("train", "val"):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        if not image_dir.is_dir():
            continue
        for image_path in image_dir.iterdir():
            if not image_path.is_file():
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"missing segment label: {label_path}")
            lines = [ln.split() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for tokens in lines:
                if len(tokens) < 7 or len(tokens) % 2 == 0:
                    raise ValueError(f"polygon needs class + >=3 points: {label_path}")
                if not 0 <= int(tokens[0]) < nc:
                    raise ValueError(f"class id out of range in {label_path}")
                values = [float(v) for v in tokens[1:]]
                if min(values) < 0.0 or max(values) > 1.0:
                    raise ValueError(f"polygon coordinates must be normalized to [0, 1]: {label_path}")
            images += 1; positives += bool(lines); polygons += len(lines)
    if images == 0 or polygons == 0:
        raise ValueError(f"no images or no polygons found from dataset YAML: {dataset_yaml}")
    return {"images": images, "positive_images": positives, "polygons": polygons}


def run_yolo_seg_training(dataset_yaml: Path, output: Path, config: dict[str, Any]) -> Path:
    """Train a YOLO instance-segmentation model.

    ``patience=0`` disables early stopping and returns ``last.pt`` (fixed-epoch protocol);
    ``patience>0`` keeps Ultralytics early stopping and returns ``best.pt``.  Mosaic and copy-paste are
    enabled: copy-paste is the main lever against the OK/NG pixel imbalance.
    """
    patience = int(config.get("patience", 0))
    if patience < 0:
        raise ValueError("patience must be >= 0")
    dataset_check = _validate_segment_dataset(dataset_yaml)
    from ultralytics import YOLO
    model = YOLO(str(config["base_checkpoint"]))
    model.train(
        data=str(dataset_yaml), task="segment", epochs=int(config["epochs"]), imgsz=int(config["imgsz"]),
        batch=int(config["batch"]), workers=int(config.get("workers", 0)), seed=int(config["seed"]),
        amp=bool(config.get("amp", True)), deterministic=bool(config.get("deterministic", True)),
        patience=patience, val=True, device=config["device"], project=str(output), name="train",
        exist_ok=False, save=True, plots=False,
        mosaic=float(config.get("mosaic", 1.0)), close_mosaic=int(config.get("close_mosaic", 10)),
        copy_paste=float(config.get("copy_paste", 0.3)), overlap_mask=True, mask_ratio=int(config.get("mask_ratio", 4)),
    )
    checkpoint = output / "train" / "weights" / ("best.pt" if patience > 0 else "last.pt")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f"training completed without a non-empty {checkpoint.name}")
    atomic_write_json(output / "training_result.json", {
        "status": "ok", "task": "segment", "checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint),
        "checkpoint_policy": "early_stopping_best_pt" if patience > 0 else "fixed_epochs_last_pt",
        "patience": patience, "epochs": int(config["epochs"]), "finished_at": utc_now(),
        "segment_dataset_check": dataset_check,
        "mosaic": float(config.get("mosaic", 1.0)), "copy_paste": float(config.get("copy_paste", 0.3)),
    })
    return checkpoint


def smoke_test_seg(checkpoint: Path, image: Path, settings: dict[str, Any], roi_mask: Path | None = None) -> dict:
    """Load the checkpoint and run one image; used before a candidate is registered."""
    from detected_pipeline.plugins.yolo_supervised import YoloSegDetector
    from detected_pipeline.roi import read_image
    detector = YoloSegDetector(checkpoint, settings, roi_mask)
    score, confs, _, masks = detector.infer(read_image(image))
    return {"ok": True, "image_score": score, "instances": int(len(confs)), "shape": list(masks.shape[1:])}
