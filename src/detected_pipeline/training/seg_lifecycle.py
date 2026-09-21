"""Milestone training, calibration and offline comparison of YOLO-seg candidates.

Data roles (never mixed):
  * training NG   confirmed NG with a valid mask, persistent hash split (default 75 %)
  * calibration   the remaining 25 % NG plus the 200 initial calibration OK reserved at initialization
  * training OK   confirmed OK from the stream and the initial memory-bank OK, batch-stratified, <= train_ok_limit
Training is fixed-epoch (``last.pt``); the calibration set doubles as the Ultralytics validation set because
no model selection happens on it.  Thresholds come from ``detected_pipeline.calibration``.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detected_pipeline.calibration import classification_metrics, mask_conf_threshold, recall_first
from detected_pipeline.masks import internal_mask
from detected_pipeline.roi import read_image, write_image
from detected_pipeline.util import atomic_write_json, sha256_file, utc_now

from .runner import run_yolo_seg_training

CONF_GRID = np.unique(np.r_[0.001, 0.002, 0.005, np.linspace(.01, .99, 99), .995, .999])


# ---------------------------------------------------------------------------------------------- data access
def confirmed_rows(workspace: Path, category: str, decision: str, include_pseudo_ok: bool = True) -> list[dict[str, Any]]:
    """NG: labeled_pool rows (confirmed NG with a valid mask).  OK: confirmed_ok_pool rows, plus the
    sampling_pseudo_ok_pool (model-OK images whose segment passed the spot check) when requested."""
    pools = ["labeled_pool"] if decision == "NG" else ["confirmed_ok_pool"] + (["sampling_pseudo_ok_pool"] if include_pseudo_ok else [])
    with closing(sqlite3.connect(workspace / "state" / "pipeline.sqlite3")) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT s.*, pm.copy_path FROM samples s JOIN pool_membership pm USING(sample_id) "
            f"WHERE s.category=? AND s.reviewed_decision=? AND pm.pool IN ({','.join('?' * len(pools))}) ORDER BY s.created_at, s.sample_id",
            (category, decision, *pools)).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        if decision == "NG":
            item["mask"] = str(workspace / "data" / category / "labeled_pool" / f"{item['sample_id']}.mask.png")
        out.append(item)
    return out


def assign_ng_splits(workspace: Path, category: str, rows: list[dict[str, Any]], calibration_fraction: float,
                     seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Persistent train/calibration role per NG: hashed once from (seed, category, sha256), stored in SQLite."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be within (0, 1)")
    train, calibration = [], []
    with closing(sqlite3.connect(workspace / "state" / "pipeline.sqlite3")) as db, db:
        for row in rows:
            old = db.execute("SELECT split FROM dataset_split_assignment WHERE sample_id=?", (row["sample_id"],)).fetchone()
            if old:
                split = old[0]
            else:
                token = f"{seed}:{category}:{row['sha256']}"
                value = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16) / float(16 ** 16)
                split = "calibration" if value < calibration_fraction else "train"
                db.execute("INSERT INTO dataset_split_assignment VALUES (?,?,?,?,?)",
                           (row["sample_id"], category, "NG", split, utc_now()))
            (calibration if split == "calibration" else train).append(dict(row, split=split))
    return train, calibration


def select_training_ok(rows: list[dict[str, Any]], excluded_sha: set[str], limit: int, seed: int) -> list[dict[str, Any]]:
    """Batch-stratified deterministic OK selection: shuffle within batch, round-robin across batches."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["sha256"] in excluded_sha:
            continue
        groups.setdefault(str(row.get("batch_id", "")), []).append(row)
    rng = random.Random(f"{seed}:ok-selection:{len(rows)}")
    for items in groups.values():
        rng.shuffle(items)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
    return selected


# ---------------------------------------------------------------------------------------------- dataset
def polygons(mask: np.ndarray) -> list[str]:
    """One normalized polygon per connected component; sub-3-point specks become their bounding box."""
    h, w = mask.shape; lines = []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        pts = contour.reshape(-1, 2).astype(np.float64)
        if len(pts) < 3:
            x, y, bw, bh = cv2.boundingRect(contour)
            pts = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], np.float64)
        pts[:, 0] = np.clip(pts[:, 0] / max(1, w - 1), 0, 1); pts[:, 1] = np.clip(pts[:, 1] / max(1, h - 1), 0, 1)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)))
    return lines


def _roi(roi_mask: Path | None, shape: tuple[int, int], cache: dict) -> np.ndarray | None:
    if roi_mask is None:
        return None
    key = shape
    if key not in cache:
        raw = read_image(roi_mask, cv2.IMREAD_GRAYSCALE)
        if raw.shape != shape:
            raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        cache[key] = raw >= 128
    return cache[key]


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def materialize(root: Path, train_items: list[tuple[Path, Path | None]], val_items: list[tuple[Path, Path | None]],
                roi_mask: Path | None) -> tuple[Path, dict[str, Any]]:
    """Write an Ultralytics segment dataset.  Items are (image, internal mask or None for OK).
    ROI categories: image outside the ROI filled white, GT ANDed with the ROI; NG whose defect lies entirely
    outside the ROI are excluded and listed in the stats."""
    cache: dict = {}; stats: dict[str, Any] = {"excluded": []}
    for split, items in (("train", train_items), ("val", val_items)):
        counts = {"images": 0, "ng": 0, "polygons": 0}
        for index, (image_path, mask_path) in enumerate(items):
            image = read_image(image_path); roi = _roi(roi_mask, image.shape[:2], cache)
            lines: list[str] = []
            if mask_path is not None:
                gt = internal_mask(mask_path, image.shape[:2])
                if roi is not None:
                    gt &= roi
                lines = polygons(gt)
                if not lines:
                    stats["excluded"].append({"split": split, "image": str(image_path), "reason": "defect entirely outside ROI"})
                    continue
            target = root / "images" / split / f"{index:05d}{image_path.suffix.lower()}"
            if roi is not None:
                out = image.copy(); out[~roi] = 255; write_image(target, out)
            else:
                _link(image_path, target)
            label = root / "labels" / split / f"{index:05d}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts["images"] += 1; counts["ng"] += bool(lines); counts["polygons"] += len(lines)
        stats[split] = counts
    import yaml
    spec = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "names": {0: "defect"}}
    (root / "data.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    atomic_write_json(root / "label_stats.json", stats)
    return root / "data.yaml", stats


# ---------------------------------------------------------------------------------------------- calibration
def calibrate_seg(detector, items: list[tuple[Path, Path | None]], rules: dict[str, Any], category: str,
                  roi_mask: Path | None = None) -> dict[str, Any]:
    """Image threshold (recall-first under the FPR cap) and mask confidence threshold (best mean NG IoU)."""
    def per_category(key: str, default: float) -> float:
        value = rules.get(key, default)
        return float(value.get(category, value.get("default", default))) if isinstance(value, dict) else float(value)
    records = []; ious: dict[float, list[float]] = {float(t): [] for t in CONF_GRID}; cache: dict = {}
    for image_path, mask_path in items:
        image = read_image(image_path)
        score, confs, _, masks = detector.infer(image)
        records.append({"image": str(image_path), "label": "NG" if mask_path else "OK", "score": score, "instances": int(len(confs))})
        if mask_path is not None:
            gt = internal_mask(mask_path, image.shape[:2]); roi = _roi(roi_mask, image.shape[:2], cache)
            if roi is not None:
                gt &= roi
            for t in CONF_GRID:
                pred = detector.union(masks, confs >= t, gt.shape); union = (pred | gt).sum()
                ious[float(t)].append(float((pred & gt).sum() / union) if union else 0.0)
    scores = [r["score"] for r in records]; labels = [r["label"] == "NG" for r in records]
    if not any(labels) or all(labels):
        raise ValueError(f"{category}: calibration needs both OK and NG images")
    image = recall_first(scores, labels, per_category("target_recall", 0.95), float(rules.get("max_fpr", 0.2)), min_threshold=0.0)
    mask = mask_conf_threshold(ious)
    return {"image_threshold": float(image["threshold"]), "mask_conf_threshold": float(mask["mask_conf_threshold"]),
            "classification": image, "segmentation": {k: v for k, v in mask.items() if k != "curve"},
            "records": records, "image_score": "max_instance_confidence"}


# ---------------------------------------------------------------------------------------------- milestones
def next_milestone(labeled_ng: int, last_milestone: int, life: dict[str, Any]) -> int | None:
    """Highest due milestone not yet trained: first at ``first_train_ng``, then +``retrain_increment``
    up to ``retrain_increment_after`` NG, then +``retrain_increment_late``."""
    first = int(life.get("first_train_ng", 40)); step = int(life.get("retrain_increment", 20))
    knee = int(life.get("retrain_increment_after", 100)); late = int(life.get("retrain_increment_late", 40))
    if labeled_ng < first:
        return None
    milestone = first
    while True:
        nxt = milestone + (step if milestone < knee else late)
        if nxt > labeled_ng:
            break
        milestone = nxt
    return milestone if milestone > last_milestone else None


def train_candidate(workspace: Path, category: str, milestone: int, config: dict[str, Any],
                    calibration_ok: list[Path], roi_mask: Path | None) -> dict[str, Any]:
    """Train one milestone candidate and calibrate it; idempotent per (milestone, data fingerprint)."""
    life = config["lifecycle"]; training = dict(config["training"]); seed = int(training["seed"])
    ng_rows = confirmed_rows(workspace, category, "NG")
    if len(ng_rows) < milestone:
        raise ValueError(f"{category}: milestone {milestone} needs {milestone} labeled NG, have {len(ng_rows)}")
    ng_rows = sorted(ng_rows, key=lambda r: (r.get("created_at", ""), r["sample_id"]))[:milestone]
    train_ng, cal_ng = assign_ng_splits(workspace, category, ng_rows, float(life.get("ng_calibration_fraction", 0.25)), int(life.get("split_seed", 42)))
    if not cal_ng:
        raise ValueError(f"{category}: no calibration NG at milestone {milestone}")
    cal_sha = {sha256_file(p) for p in calibration_ok}
    ok_rows = select_training_ok(confirmed_rows(workspace, category, "OK", bool(life.get("pseudo_ok_in_training", True))), cal_sha, int(life.get("train_ok_limit", 400)), seed)
    if len(ok_rows) < int(life.get("min_train_ok", 100)):
        raise ValueError(f"{category}: only {len(ok_rows)} training OK available")
    identity = {"train_ng": sorted(r["sha256"] for r in train_ng), "cal_ng": sorted(r["sha256"] for r in cal_ng),
                "train_ok": sorted(r["sha256"] for r in ok_rows), "cal_ok": sorted(cal_sha), "training": training,
                "thresholds": life.get("yolo_thresholds", {})}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    run_root = workspace / "model_registry" / category / "milestones" / f"v{milestone}_{fingerprint}"
    summary_path = run_root / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    train_items = [(Path(r["copy_path"]), Path(r["mask"])) for r in train_ng] + [(Path(r["copy_path"]), None) for r in ok_rows]
    cal_items = [(Path(p), None) for p in calibration_ok] + [(Path(r["copy_path"]), Path(r["mask"])) for r in cal_ng]
    dataset_root = run_root / "dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    yaml_path, stats = materialize(dataset_root, train_items, cal_items, roi_mask)
    started = time.perf_counter()
    training["base_checkpoint"] = str(training["base_checkpoint"]); training["patience"] = 0
    checkpoint = run_yolo_seg_training(yaml_path, run_root / "model", training)
    training_seconds = time.perf_counter() - started
    from detected_pipeline.plugins.yolo_supervised import YoloSegDetector
    detector = YoloSegDetector(checkpoint, config["training"].get("inference", {}), roi_mask)
    excluded = {Path(e["image"]) for e in stats["excluded"]}
    calibration = calibrate_seg(detector, [it for it in cal_items if it[0] not in excluded], life.get("yolo_thresholds", {}), category, roi_mask)
    del detector
    summary = {"status": "candidate", "category": category, "milestone": milestone, "fingerprint": fingerprint,
               "model_version": f"{category}-seg-v{milestone}-{fingerprint}", "checkpoint": str(checkpoint),
               "thresholds": {"image_threshold": calibration["image_threshold"], "mask_conf_threshold": calibration["mask_conf_threshold"]},
               "calibration": {k: v for k, v in calibration.items() if k != "records"},
               "calibration_records": calibration["records"],
               "counts": {"train_ng": len(train_ng), "train_ok": len(ok_rows), "train_pseudo_ok": int(sum(r.get("label_source") == "sampling_pseudo_ok" for r in ok_rows)), "calibration_ng": len(cal_ng),
                          "calibration_ok": len(calibration_ok), "excluded_outside_roi": len(stats["excluded"])},
               "dataset_stats": stats, "training_seconds": training_seconds, "created_at": utc_now()}
    atomic_write_json(summary_path, summary)
    shutil.rmtree(dataset_root, ignore_errors=True)   # generated links only; source data untouched
    return summary


def compare_models(official_scores: list[float], official_threshold: float, candidate_scores: list[float],
                   candidate_threshold: float, labels: list[bool], gate: dict[str, Any]) -> dict[str, Any]:
    """Offline promotion gate on the calibration set (recall-first): the candidate must not add misses and
    must not raise the OK false-positive rate; it must improve at least one of the two meaningfully."""
    old = classification_metrics(labels, [s >= official_threshold for s in official_scores])
    new = classification_metrics(labels, [s >= candidate_threshold for s in candidate_scores])
    max_fpr_increase = float(gate.get("max_fpr_increase", 0.0)); min_fpr_gain = float(gate.get("min_fpr_reduction_for_equal_fn", 0.01))
    checks = {"no_extra_misses": new["fn"] <= old["fn"],
              "fpr_not_worse": new["ok_false_positive_rate"] <= old["ok_false_positive_rate"] + max_fpr_increase,
              "real_gain": new["fn"] < old["fn"] or new["ok_false_positive_rate"] <= old["ok_false_positive_rate"] - min_fpr_gain}
    return {"decision": "promote" if all(checks.values()) else "reject", "checks": checks, "official": old, "candidate": new,
            "official_threshold": official_threshold, "candidate_threshold": candidate_threshold, "compared_at": utc_now()}
