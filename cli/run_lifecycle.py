"""Per-category closed loop: pretrained detector -> sampled review -> YOLO-seg milestones -> shadow -> promotion.

  0 NG                 pretrained official (OK-quantile threshold, hybrid segmentation)
  every batch          model-NG fully reviewed; model-OK spot-checked by score segment (top 20 % full, middle 40 %
                       at 10 %, low 40 % at 2 %; an NG in a sample escalates the segment); unreviewed = pseudo OK
  every 5 NG           pretrained threshold re-selected (ladder < 30 NG, recall-first >= 30)
  40 NG, +20, +40      YOLO-seg candidate: offline gate on the calibration set -> shadow from the next batch ->
                       judged after >= shadow_min_ng reviewed NG (or shadow_max_batches) -> promote or reject
  every model          evaluated on the fixed test set (report only)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detected_pipeline.cache_identity import fingerprint
from detected_pipeline.config import load_project_config, roi_mask_for
from detected_pipeline.contracts import Decision, InferenceContext, ReviewRecord
from detected_pipeline.evaluation import fixed_test_metrics, score_fixed_test, write_test_report
from detected_pipeline.experiment_observation import batch_snapshot
from detected_pipeline.feedback import FeedbackStore
from detected_pipeline.online_metrics import reviewed_metrics
from detected_pipeline.plugins import load_pretrained_plugin
from detected_pipeline.plugins.yolo_supervised import YoloFeedbackAdapter, YoloSegDetector
from detected_pipeline.registry import ModelRegistry
from detected_pipeline.review import build_review_provider, run_sampled_review
from detected_pipeline.roi import read_image
from detected_pipeline.training.runner import smoke_test_seg
from detected_pipeline.training.seg_lifecycle import compare_models, confirmed_rows, next_milestone, train_candidate
from detected_pipeline.util import atomic_write_json, sha256_file, utc_now

KEYS = ("recall", "ok_false_positive_rate", "test_auroc", "iou_micro", "mean_iou_all_ng")


def per_category(value, category, default):
    if isinstance(value, dict):
        return value.get(category, value.get("default", default))
    return default if value is None else value


def random_stream(paths, seed):
    ordered = list(paths); random.Random(seed).shuffle(ordered); return ordered


def evenly_mixed_stream(paths, batch_size, seed):
    """Deterministically stratify OK/NG so every review batch has the same NG rate."""
    ok = [p for p in paths if p.parent.name.upper() == "OK"]; ng = [p for p in paths if p.parent.name.upper() == "NG"]
    other = [p for p in paths if p.parent.name.upper() not in {"OK", "NG"}]
    rng = random.Random(seed); rng.shuffle(ok); rng.shuffle(ng); rng.shuffle(other)
    total = len(ok) + len(ng)
    if not total:
        return other
    buckets = [[] for _ in range((total + batch_size - 1) // batch_size)]
    for index, path in enumerate(ng):
        buckets[index % len(buckets)].append(path)
    ok_index = 0
    for bucket in buckets:
        take = min(batch_size - len(bucket), len(ok) - ok_index); bucket.extend(ok[ok_index:ok_index + take]); ok_index += take
    while ok_index < len(ok):
        for bucket in buckets:
            if ok_index >= len(ok):
                break
            if len(bucket) < batch_size:
                bucket.append(ok[ok_index]); ok_index += 1
    ordered = []
    for index, bucket in enumerate(buckets):
        random.Random(f"{seed}:batch:{index}").shuffle(bucket); ordered.extend(bucket)
    return ordered + other


def yolo_adapter(model, workspace, config, tag, roi_mask):
    plugin = YoloFeedbackAdapter(Path(model["checkpoint"]), {
        **model["thresholds"], **config["training"].get("inference", {}), "model_version": model["model_version"],
        "output_dir": str(workspace / "inference_results" / tag), "roi_mask": str(roi_mask) if roi_mask else None})
    plugin.load(); return plugin


def initialization(manifest_path: Path, category: str):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next((r for r in manifest["classes"] if r["category"] == category), None)
    if row is None:
        raise KeyError(f"initialization manifest has no category {category}")
    return ([Path(p) for p in row["initial_reference_ok"]], [Path(p) for p in row["initial_bank_ok"]],
            [Path(p) for p in row["initial_calibration_ok"]], list(row.get("fixed_test", [])), manifest.get("stream_mode", "stratified"))


def main():
    p = argparse.ArgumentParser(description="pretrained -> YOLO-seg closed loop for one category")
    p.add_argument("--category", required=True); p.add_argument("--inbox", type=Path, required=True)
    p.add_argument("--initialization-manifest", type=Path, required=True); p.add_argument("--batch-id", default="lifecycle")
    args = p.parse_args()
    config = load_project_config(PROJECT); categories = list(config["categories"]); category = args.category
    if category not in categories:
        p.error(f"unknown category: {category}")
    life = dict(config["lifecycle"]); workspace = Path(config["workspace_root"]); roi_mask = roi_mask_for(config, category)
    for key in ("first_train_ng", "retrain_increment", "retrain_increment_after", "retrain_increment_late"):
        life[key] = per_category(life.get(key), category, life.get(key))
    sampling = dict(config.get("review_sampling", {}))
    store = FeedbackStore(workspace, categories); reviewer = build_review_provider(config); registry = ModelRegistry(workspace)
    state_path = workspace / "state" / f"lifecycle_{category}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "category": category, "reviewed_batches": 0, "last_milestone": 0, "last_pretrained_calibration_ng": -1,
        "candidate": None, "shadow_rows": [], "shadow_batches": 0, "history": []}
    reference_ok, bank_ok, calibration_ok, fixed_test, stream_mode = initialization(args.initialization_manifest, category)
    calibration_sha = {sha256_file(path) for path in calibration_ok}
    for image in bank_ok:   # memory-bank OK are also permanent YOLO training OK (never stream, never calibration)
        store.seed_confirmed_ok_for_training(image, category, f"{category}-{sha256_file(image)[:16]}")

    suffixes = {s.lower() for s in config["allowed_image_suffixes"]}
    paths = [x for x in args.inbox.rglob("*") if x.is_file() and x.suffix.lower() in suffixes and not reviewer.is_ground_truth_artifact(x)]
    batch_size = int(per_category(config["review_batch_size"], category, 200))
    stream_seed = f"{config.get('simulation_stream_seed', 42)}:{category}:stream"
    paths = random_stream(paths, stream_seed) if stream_mode == "random" else evenly_mixed_stream(paths, batch_size, stream_seed)
    paths = [x for x in paths if not store.contains(category, sha256_file(x))]

    pretrained = load_pretrained_plugin(Path(config["pretrained_config"]), PROJECT)
    pretrained.prepare_category(category, reference_ok + bank_ok, calibration_ok)
    test_cache = workspace / "test_cache" / category

    # ------------------------------------------------------------------ fixed test set helpers
    visuals = bool(config.get("fixed_test_visuals", True))

    def pretrained_predict(path):
        from detected_pipeline.pretrained import hybrid_regions
        heat, valid = pretrained.detector.heatmap(category, read_image(path)); score = pretrained.detector.image_score(heat, valid)
        seg = pretrained.segmentation
        mask, stats = hybrid_regions(heat, valid, float(pretrained.thresholds[category]["pixel_threshold"]), float(seg.get("peak_fraction", 0.7)),
                                     int(seg.get("min_area", 128)), int(seg.get("max_regions", 3)))
        boxes = [[*r["bbox_xyxy"], f"#{r['rank']} {r['peak']:.2f}"] for r in stats["regions"]]
        return {"score": score, "mask": mask, "heat": heat, "boxes": boxes, "extra": {"regions": stats["regions"]}}

    def yolo_predict_fn(detector, mask_threshold):
        def predict(path):
            image = read_image(path); score, confs, boxes, masks = detector.infer(image)
            keep = confs >= mask_threshold
            heat = np.zeros(image.shape[:2], np.float32)
            for m, c in zip(masks, confs):
                heat[m] = np.maximum(heat[m], c)
            kept_boxes = [[*b, f"{c:.2f}"] for b, c in zip(boxes[keep], confs[keep])]
            instances = [{"confidence": float(c), "bbox_xyxy": [float(v) for v in b], "kept": bool(k)} for c, b, k in zip(confs, boxes, keep)]
            return {"score": score, "mask": detector.union(masks, keep, image.shape[:2]), "heat": heat, "boxes": kept_boxes,
                    "extra": {"mask_conf_threshold": float(mask_threshold), "instances": sorted(instances, key=lambda z: -z["confidence"])[:20]}}
        return predict

    def test_pretrained(role):
        if not fixed_test:
            return None
        key = "pretrained-" + fingerprint({"bank": pretrained.detector.categories[category]["manifest"]["fingerprint"],
                                           "segmentation": pretrained.segmentation,
                                           "pixel_threshold": pretrained.thresholds[category]["pixel_threshold"],
                                           "top_fraction": pretrained.detector.top_fraction})
        cache = test_cache / key
        rows = score_fixed_test(fixed_test, pretrained_predict, cache, key, roi_mask, visuals, lambda m: print(m, flush=True))
        thresholds = pretrained.thresholds[category]
        write_test_report(workspace, category, key, role, rows, float(thresholds["image_threshold"]), cache,
                          {"threshold_rule": thresholds["rule"], "calibration_ng": thresholds.get("calibration_ng", 0)})
        return fixed_test_metrics(rows, float(thresholds["image_threshold"]))

    def test_yolo(model, role):
        if not fixed_test:
            return None
        settings = config["training"].get("inference", {})
        key = "yolo-" + fingerprint({"checkpoint": sha256_file(Path(model["checkpoint"])), "inference": settings,
                                     "mask_conf_threshold": model["thresholds"]["mask_conf_threshold"], "schema": 2})
        cache = test_cache / key
        detector = None
        def predict(path):
            nonlocal detector
            if detector is None:
                detector = YoloSegDetector(Path(model["checkpoint"]), settings, roi_mask)
            return yolo_predict_fn(detector, float(model["thresholds"]["mask_conf_threshold"]))(path)
        rows = score_fixed_test(fixed_test, predict, cache, key, roi_mask, visuals, lambda m: print(m, flush=True))
        del detector
        write_test_report(workspace, category, model["model_version"], role, rows, float(model["thresholds"]["image_threshold"]), cache)
        return fixed_test_metrics(rows, float(model["thresholds"]["image_threshold"]))

    # ------------------------------------------------------------------ pretrained threshold maintenance
    def confirmed_ng_paths():
        return [Path(r["copy_path"]) for r in confirmed_rows(workspace, category, "NG")]

    def recalibrate_pretrained(force=False):
        n = len(confirmed_ng_paths()); every = int(life.get("pretrained_recalibrate_every_ng", 5))
        if force or n // every > state["last_pretrained_calibration_ng"] // every:
            result = pretrained.calibrate(category, confirmed_ng_paths()); state["last_pretrained_calibration_ng"] = n
            test = test_pretrained("production")
            state["history"].append({"event": "pretrained_recalibrated", "at": utc_now(), "rule": result["rule"], "image_threshold": result["image_threshold"],
                                     "calibration_ng": result["calibration_ng"], "fixed_test": {k: test[k] for k in KEYS} if test else None})

    production = registry.current(category)
    official = yolo_adapter(production, workspace, config, "production_yolo", roi_mask) if production else pretrained
    if official is pretrained:
        recalibrate_pretrained(force=state["last_pretrained_calibration_ng"] < 0)
    candidate = yolo_adapter(state["candidate"], workspace, config, "shadow_yolo", roi_mask) if state.get("candidate") else None

    def review_record(prediction, image, pseudo_ok=False):
        if pseudo_ok:
            return ReviewRecord(sample_id=prediction.sample_id, category=category, model_decision=prediction.final_decision,
                                reviewed_decision=Decision.OK, label_source="sampling_pseudo_ok", mask_path=None, reviewer="sampling_pseudo_ok",
                                metadata={"source_path": str(image)})
        return reviewer.review_prediction(prediction, image)

    try:
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]; pending = []
            for image in batch:
                digest = sha256_file(image); sample_id = f"{category}-{digest[:16]}"
                context = InferenceContext(sample_id, category, f"{args.batch_id}-{state['reviewed_batches'] + 1:04d}")
                prediction = official.predict(image, context)
                if not store.ingest(image, context, prediction):
                    continue
                shadow_prediction = candidate.predict(image, context) if candidate is not None else None
                pending.append((image, prediction, shadow_prediction))

            # ---- review: model-NG fully, model-OK by score segment; the rest become pseudo OK
            rows = [{"sample_id": pr.sample_id, "image": str(im), "official": pr.final_decision.value, "official_score": pr.feature_norm_score,
                     "official_mask": pr.binary_mask_path, "truth": None, "review": None,
                     **({"shadow": sh.final_decision.value, "shadow_score": sh.feature_norm_score, "shadow_mask": sh.binary_mask_path} if sh else {})}
                    for im, pr, sh in pending]
            def review_index(i):
                image, prediction, _ = pending[i]; record = review_record(prediction, image); store.apply_review(image, record)
                rows[i].update(truth=record.reviewed_decision.value, review="reviewed", gt_mask=record.mask_path,
                               label_source=record.label_source); return record.reviewed_decision == Decision.NG
            ng_indices = [i for i, (_, pr, _) in enumerate(pending) if pr.final_decision == Decision.NG]
            ok_indices = [i for i, (_, pr, _) in enumerate(pending) if pr.final_decision == Decision.OK]
            for i in ng_indices:
                review_index(i)
            sampling_stats = None
            if sampling.get("enabled", True) and ok_indices:
                outcome = run_sampled_review([pending[i][1].feature_norm_score for i in ok_indices], sampling,
                                             f"{stream_seed}:review:{state['reviewed_batches'] + 1}", lambda j: review_index(ok_indices[j]))
                sampling_stats = outcome["stats"]; hidden_ng = 0
                for j in outcome["pseudo_ok"]:
                    i = ok_indices[j]; image, prediction, _ = pending[i]
                    store.apply_review(image, review_record(prediction, image, pseudo_ok=True))
                    hidden = reviewer.review_prediction(prediction, image).reviewed_decision.value   # simulation only: never used for decisions
                    rows[i].update(review="pseudo_ok", hidden_truth=hidden); hidden_ng += hidden == "NG"
                sampling_stats["pseudo_ok"] = len(outcome["pseudo_ok"]); sampling_stats["pseudo_ok_hidden_ng"] = hidden_ng
            else:
                for i in ok_indices:
                    review_index(i)
            state["reviewed_batches"] += 1
            evaluated = reviewed_metrics(rows, roi_mask=roi_mask)
            report = {"at": utc_now(), "batch": state["reviewed_batches"], "count": len(rows), "reviewed": sum(r["review"] == "reviewed" for r in rows),
                      "official_model": production["model_version"] if official is not pretrained else "pretrained",
                      "official": evaluated["classification"], "official_segmentation": evaluated["segmentation"],
                      "review_sampling": sampling_stats, "rows": rows}
            if candidate is not None:
                shadow_rows = [r for r in rows if "shadow" in r and r["review"] == "reviewed"]
                shadow_evaluated = reviewed_metrics(shadow_rows, "shadow", "shadow_mask", roi_mask)
                report["shadow"] = shadow_evaluated["classification"]
                report["shadow_segmentation"] = shadow_evaluated["segmentation"]
                report["shadow_model"] = state["candidate"]["model_version"]
                report["disagreements"] = [r["sample_id"] for r in shadow_rows if r["shadow"] != r["official"]]
                state["shadow_rows"].extend({k: r[k] for k in ("sample_id", "truth", "official", "shadow")} for r in shadow_rows)
                state["shadow_batches"] += 1
            atomic_write_json(workspace / "batch_reports" / category / f"batch_{state['reviewed_batches']:04d}.json", report)
            atomic_write_json(state_path, state)

            # ---- shadow judgement (same gate as offline, on reviewed shadow rows)
            if candidate is not None:
                shadow_ng = sum(r["truth"] == "NG" for r in state["shadow_rows"])
                if shadow_ng >= int(life.get("shadow_min_ng", 10)) or state["shadow_batches"] >= int(life.get("shadow_max_batches", 5)):
                    labels = [r["truth"] == "NG" for r in state["shadow_rows"]]
                    verdict = compare_models([1.0 if r["official"] == "NG" else 0.0 for r in state["shadow_rows"]], 0.5,
                                             [1.0 if r["shadow"] == "NG" else 0.0 for r in state["shadow_rows"]], 0.5, labels, life.get("promotion", {}))
                    verdict.update(stage="shadow", reviewed=len(labels), ng=shadow_ng, batches=state["shadow_batches"], model_version=state["candidate"]["model_version"])
                    atomic_write_json(workspace / "promotion_reports" / category / f"shadow_{state['candidate']['model_version']}.json", verdict)
                    meta_path = registry.root / category / "versions" / state["candidate"]["model_version"] / "model.json"
                    metadata = json.loads(meta_path.read_text(encoding="utf-8")); metadata["shadow_comparison"] = verdict
                    if verdict["decision"] == "promote":
                        metadata["status"] = "promoted"; atomic_write_json(meta_path, metadata)
                        production = registry.promote(category, metadata)
                        if official is not pretrained:
                            official.close()
                        candidate.close(); candidate = None
                        official = yolo_adapter(production, workspace, config, "production_yolo", roi_mask)
                        test_yolo(metadata, "production")
                        state["history"].append({"event": "promoted", "at": utc_now(), "model": production["model_version"], "shadow_ng": shadow_ng})
                    else:
                        metadata["status"] = "rejected_shadow"; atomic_write_json(meta_path, metadata)
                        candidate.close(); candidate = None
                        state["history"].append({"event": "rejected_shadow", "at": utc_now(), "model": state["candidate"]["model_version"], "checks": verdict["checks"]})
                    state["candidate"] = None; state["shadow_rows"] = []; state["shadow_batches"] = 0
                    atomic_write_json(state_path, state)

            if official is pretrained:
                recalibrate_pretrained()

            # ---- YOLO milestone: train, offline gate, start shadow
            labeled_ng = store.counts(category)["labeled_pool"]
            milestone = next_milestone(labeled_ng, state["last_milestone"], life)
            if milestone:
                if official is not pretrained:
                    official.close()
                if candidate is not None:   # a newer milestone supersedes the candidate still in shadow
                    candidate.close(); candidate = None
                    meta_path = registry.root / category / "versions" / state["candidate"]["model_version"] / "model.json"
                    superseded = json.loads(meta_path.read_text(encoding="utf-8")); superseded["status"] = "superseded"; atomic_write_json(meta_path, superseded)
                    state["history"].append({"event": "superseded", "at": utc_now(), "model": state["candidate"]["model_version"]})
                    state["candidate"] = None; state["shadow_rows"] = []; state["shadow_batches"] = 0
                summary = train_candidate(workspace, category, milestone, config, calibration_ok, roi_mask)
                smoke_image = Path(summary["calibration_records"][0]["image"])
                metadata = registry.register_candidate(category, summary["model_version"], Path(summary["checkpoint"]), f"milestone-v{milestone}",
                                                       summary["thresholds"], lambda path: smoke_test_seg(path, smoke_image, config["training"].get("inference", {}), roi_mask))
                metadata.update({"milestone": milestone, "calibration": summary["calibration"], "counts": summary["counts"]})
                cal_items = [(Path(r["image"]), r["label"] == "NG") for r in summary["calibration_records"]]
                candidate_scores = [float(r["score"]) for r in summary["calibration_records"]]
                if official is pretrained:
                    official_scores = [pretrained.score(category, path) for path, _ in cal_items]; official_threshold = float(pretrained.thresholds[category]["image_threshold"])
                else:
                    scorer = YoloSegDetector(Path(production["checkpoint"]), config["training"].get("inference", {}), roi_mask)
                    official_scores = [scorer.score(path) for path, _ in cal_items]; official_threshold = float(production["thresholds"]["image_threshold"]); del scorer
                comparison = compare_models(official_scores, official_threshold, candidate_scores, float(summary["thresholds"]["image_threshold"]),
                                            [is_ng for _, is_ng in cal_items], life.get("promotion", {}))
                comparison["stage"] = "offline"; metadata["offline_comparison"] = comparison
                metadata["status"] = "shadow_candidate" if comparison["decision"] == "promote" else "rejected_offline"
                metadata["fixed_test"] = test_yolo(metadata, "candidate")
                atomic_write_json(registry.root / category / "versions" / summary["model_version"] / "model.json", metadata)
                atomic_write_json(workspace / "promotion_reports" / category / f"offline_{summary['model_version']}.json", {**comparison, "model_version": summary["model_version"]})
                state["last_milestone"] = milestone
                state["history"].append({"event": "candidate_trained", "at": utc_now(), "model": summary["model_version"], "milestone": milestone, "labeled_ng": labeled_ng,
                                         "offline_decision": comparison["decision"], "fixed_test": {k: metadata["fixed_test"][k] for k in KEYS} if metadata.get("fixed_test") else None})
                if comparison["decision"] == "promote":
                    state["candidate"] = metadata; state["shadow_rows"] = []; state["shadow_batches"] = 0
                    candidate = yolo_adapter(metadata, workspace, config, "shadow_yolo", roi_mask)
                if official is not pretrained:
                    official = yolo_adapter(production, workspace, config, "production_yolo", roi_mask)
                atomic_write_json(state_path, state)
            report["end_of_batch"] = batch_snapshot(workspace, category, state, config, calibration_sha, roi_mask)
            atomic_write_json(workspace / "batch_reports" / category / f"batch_{state['reviewed_batches']:04d}.json", report)
            print(json.dumps({"batch": state["reviewed_batches"], "processed": len(pending), "reviewed": report["reviewed"], "labeled_ng": labeled_ng,
                              "official": production["model_version"] if official is not pretrained else "pretrained",
                              "shadow": state["candidate"]["model_version"] if state.get("candidate") else None, "last_milestone": state["last_milestone"]}, ensure_ascii=False), flush=True)
    finally:
        if official is not pretrained:
            official.close()
        if candidate is not None:
            candidate.close()
        pretrained.close()
    print(json.dumps({"status": "complete", "category": category, "state": str(state_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
