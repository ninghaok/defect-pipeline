"""Synthetic counterexamples for reporting; no GPU or real dataset is required."""
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from detected_pipeline.evaluation import fixed_test_metrics, score_fixed_test, write_test_report
from detected_pipeline.feedback import FeedbackStore
from detected_pipeline.online_metrics import reviewed_metrics
from detected_pipeline.plugins.yolo_supervised import YoloSegDetector
from detected_pipeline.roi import read_image, write_image
from detected_pipeline.util import atomic_write_json, sha256_file


def cli(name):
    path = Path(__file__).resolve().parents[1] / "cli"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(root, name, gt, pred=None, score=.9):
    image, mask, prediction = root / f"{name}.png", root / f"{name}_gt.png", root / f"{name}_pred.png"
    write_image(image, np.full((*gt.shape, 3), 80, np.uint8))
    write_image(mask, (~gt).astype(np.uint8) * 255)
    write_image(prediction, (gt if pred is None else pred).astype(np.uint8) * 255)
    item = {"image": str(image), "label": "NG", "mask": str(mask)}
    online = {"image": str(image), "truth": "NG", "gt_mask": str(mask), "official_mask": str(prediction),
              "official": "NG" if score >= .5 else "OK", "label_source": "folder_ground_truth"}
    return item, online


def test_micro_weights_pixels_and_classification_misses_keep_gt_union(tmp_path):
    small = np.zeros((12, 12), bool); small[0, 0] = True
    large = np.zeros_like(small); large[:10, :10] = True
    a, ar = sample(tmp_path, "small", small)
    b, br = sample(tmp_path, "large", large, score=.1)
    # Raw masks are both perfect, but the large NG is classified OK.
    rows = score_fixed_test([a, b], lambda p: {"score": .9 if p.stem == "small" else .1,
                                            "mask": small if p.stem == "small" else large}, tmp_path / "cache", "model", visuals=False)
    metrics = fixed_test_metrics(rows, .5)
    online = reviewed_metrics([ar, br])["segmentation"]
    assert metrics["iou_micro"] == pytest.approx(1 / 101) == online["iou_micro"]
    assert metrics["mean_iou_all_ng"] == .5
    assert metrics["recall"] == .5
    assert fixed_test_metrics(rows, .05)["iou_micro"] == 1
    report = write_test_report(tmp_path, "cat", "model", "test", rows, .5)
    case = json.loads((report / "fn" / "00001" / "score.json").read_text())
    assert case["raw_localization_iou"] == 1 and case["end_to_end_iou"] == 0
    assert case["intersection"] == 0 and case["union"] == 100


def test_roi_clips_both_masks_and_excludes_outside_only_ng(tmp_path):
    gt = np.zeros((4, 4), bool); gt[0, 0] = gt[3, 3] = True
    pred = gt.copy(); pred[3, 2] = True  # outside-ROI false foreground must not lower IoU
    item, online = sample(tmp_path, "mixed", gt, pred)
    outside = np.zeros_like(gt); outside[3, 3] = True
    outside_item, outside_online = sample(tmp_path, "outside", outside, score=.1)
    roi = tmp_path / "roi.png"; selected = np.zeros_like(gt); selected[:2, :2] = True
    write_image(roi, selected.astype(np.uint8) * 255)
    rows = score_fixed_test([item, outside_item], lambda p: {"score": .9 if p.stem == "mixed" else .1, "mask": pred},
                            tmp_path / "cache", "model", roi, False)
    fixed = fixed_test_metrics(rows, .5)
    live = reviewed_metrics([online, outside_online], roi_mask=roi)
    assert fixed["iou_micro"] == live["segmentation"]["iou_micro"] == 1
    assert fixed["recall"] == live["classification"]["recall"] == 1
    assert fixed["excluded_outside_roi"] == live["classification"]["excluded_outside_roi"] == 1
    assert fixed["test_ng"] == 1


@pytest.mark.parametrize("kind", ["missing", "unreadable", "zero_bytes", "empty", "wrong_shape"])
def test_invalid_gt_does_not_become_zero_iou(tmp_path, kind):
    gt = np.ones((4, 4), bool)
    a, ar = sample(tmp_path, "valid", gt)
    b, br = sample(tmp_path, "invalid", gt)
    mask = Path(b["mask"])
    if kind == "missing": mask.unlink()
    elif kind == "unreadable": mask.write_bytes(b"not an image")
    elif kind == "zero_bytes": mask.write_bytes(b"")
    elif kind == "empty": write_image(mask, np.full((4, 4), 255, np.uint8))
    else: write_image(mask, np.zeros((2, 2), np.uint8))
    rows = score_fixed_test([a, b], lambda p: {"score": .9, "mask": gt}, tmp_path / "cache", "model", visuals=False)
    result = fixed_test_metrics(rows, .5)
    live = reviewed_metrics([ar, br])["segmentation"]
    assert result["recall"] == 1  # image label remains available without ROI
    report = write_test_report(tmp_path, "cat", "model", "test", rows, .5)
    saved = json.loads((report / "report.json").read_text())
    assert len(saved["gt_issues"]) == 1 and saved["iou_micro"] is None
    for metrics in (result, live):
        assert metrics["iou_micro"] is None and metrics["mean_iou_all_ng"] is None
        assert metrics["segmentation_valid_ng"] == metrics["segmentation_invalid_ng"] == 1
    # ROI membership cannot be established for the invalid annotation.
    roi = tmp_path / "roi.png"; write_image(roi, np.full((4, 4), 255, np.uint8))
    roi_rows = score_fixed_test([a, b], lambda p: {"score": .9, "mask": gt}, tmp_path / "cache", "model", roi, False)
    assert fixed_test_metrics(roi_rows, .5)["excluded_invalid_gt"] == 1


def test_summary_excludes_initial_pseudo_assumed_and_pending_records(tmp_path):
    report = cli("report_metrics")
    ws = tmp_path / "ws"; store = FeedbackStore(ws, ["cat"])
    gt = np.ones((4, 4), bool)
    decisions = [("NG", "NG", "folder_ground_truth"), ("OK", "NG", "human_review"),
                 ("NG", "OK", "folder_ground_truth"), ("OK", "OK", "human_review"),
                 ("OK", "OK", "sampling_pseudo_ok"), ("OK", "OK", "model_assumed_review"),
                 ("OK", "OK", "initial_calibration_yolo_train_ok"), ("OK", None, None)]
    for index, (prediction, truth, source) in enumerate(decisions):
        item, record = sample(tmp_path, str(index), gt)
        # Human annotations use internal (white-defect) polarity.
        if source == "human_review": write_image(Path(item["mask"]), gt.astype(np.uint8) * 255)
        with sqlite3.connect(store.db_path) as db:
            db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                str(index), str(index), "cat", item["image"], "batch", "initialization" if index == 6 else "camera",
                prediction, truth, source, item["mask"] if truth == "NG" else None, "now"))
        atomic_write_json(ws / "inference_results" / "cat" / f"{index}.json",
                          {"binary_mask_path": record["official_mask"], "latency_ms": 1})
    result = report.category_metrics(ws, "cat")
    assert [result[k] for k in ("tp", "fn", "fp", "tn")] == [1, 1, 1, 1]
    assert result["count"] == 4 and result["recall"] == result["ok_false_positive_rate"] == .5
    assert result["scope"] == "reviewed_online_subset"
    assert result["excluded_initialization"] == 1 and result["excluded_unverified"] == 3
    assert result["iou_micro"] == .5


@pytest.mark.parametrize("change", ["image_path", "image_bytes", "mask_bytes", "label", "roi", "inference", "model"])
def test_cache_invalidates_changed_inputs_with_same_item_count(tmp_path, change):
    gt = np.ones((4, 4), bool)
    item, _ = sample(tmp_path, "a", gt)
    calls = []
    def predict(path):
        calls.append(path)
        return {"score": .9, "mask": gt}
    roi = tmp_path / "roi.png"; write_image(roi, np.full((4, 4), 255, np.uint8))
    cache = tmp_path / "cache"; settings = {"mask_threshold": .3}; model = "model"
    score_fixed_test([item], predict, cache, model, roi, False, inference_settings=settings)
    assert len(calls) == 1
    assert score_fixed_test([item], None, cache, model, roi, False, inference_settings=settings)
    if change == "image_path":
        other, _ = sample(tmp_path, "b", gt); item = other
    elif change == "image_bytes": write_image(Path(item["image"]), np.full((4, 4, 3), 81, np.uint8))
    elif change == "mask_bytes":
        modified = np.zeros((4, 4), np.uint8); modified[0, 0] = 255; write_image(Path(item["mask"]), modified)
    elif change == "label": item = {**item, "label": "OK"}
    elif change == "roi":
        modified = np.full((4, 4), 255, np.uint8); modified[0] = 0; write_image(roi, modified)
    elif change == "inference": settings = {"mask_threshold": .8}
    else: model = "other-model"
    with pytest.raises(RuntimeError, match="no valid fixed-test cache"):
        score_fixed_test([item], None, cache, model, roi, False, inference_settings=settings)
    rows = score_fixed_test([item], predict, cache, model, roi, False, inference_settings=settings)
    assert len(calls) == 2 and rows[0]["image"] == item["image"]


def test_cache_rebuild_does_not_modify_historical_linked_visuals(tmp_path):
    gt = np.ones((4, 4), bool); item, _ = sample(tmp_path, "a", gt)
    cache = tmp_path / "cache"
    rows = score_fixed_test([item], lambda p: {"score": .9, "mask": gt}, cache, "model-a")
    report = write_test_report(tmp_path, "cat", "a", "test", rows, .5, cache)
    archived_mask = report / "tp" / "00000" / "pred_mask.png"
    original = sha256_file(archived_mask)
    score_fixed_test([item], lambda p: {"score": .9, "mask": np.zeros_like(gt)}, cache, "model-b")
    assert sha256_file(archived_mask) == original


def test_legacy_cache_and_missing_visuals_are_not_reused(tmp_path):
    gt = np.ones((4, 4), bool); item, _ = sample(tmp_path, "a", gt)
    cache = tmp_path / "cache"; calls = []
    atomic_write_json(cache / "scores.json", {"model_key": "model", "rows": [{"score": 999}]})
    def predict(path):
        calls.append(path)
        return {"score": .9, "mask": gt}
    rows = score_fixed_test([item], predict, cache, "model", visuals=False)
    assert len(calls) == 1 and rows[0]["score"] == .9
    rows = score_fixed_test([item], predict, cache, "model", visuals=True)
    assert len(calls) == 2 and (Path(rows[0]["visuals_dir"]) / "boxed.jpg").is_file()


@pytest.mark.parametrize("changed", ["bank", "top_fraction", "pixel_quantile", "rules", "ng_bytes"])
def test_pretrained_calibration_cache_tracks_dependencies(tmp_path, changed):
    from detected_pipeline.plugins.pretrained_patchcore import PretrainedPatchCorePlugin
    class Detector:
        top_fraction = .005
        bank_key = "weights-and-bank-a"
        def __init__(self): self.categories = {}; self.calls = 0
        def build(self, category, *args):
            state = {"manifest": {"fingerprint": self.bank_key}}
            self.categories[category] = state
            return state
        def score_image(self, category, path):
            self.calls += 1
            return .8 if path.stem == "ng" else .2
        def ok_pixel_sample(self, *args): return np.arange(100, dtype=float)
    ok = tmp_path / "ok.png"; ng = tmp_path / "ng.png"
    for path in (ok, ng): write_image(path, np.zeros((4, 4, 3), np.uint8))
    plugin = PretrainedPatchCorePlugin(); plugin.detector = Detector()
    plugin.cache_root = tmp_path / "cache"; plugin.roi_masks = {}
    plugin.segmentation = {}; plugin.rules = {"min_calibration_ok": 1}
    plugin.prepare_category("cat", [ok], [ok]); plugin.calibrate("cat", [ng])
    count = plugin.detector.calls
    plugin.prepare_category("cat", [ok], [ok])
    assert plugin.detector.calls == count
    if changed == "bank": plugin.detector.bank_key = "weights-and-bank-b"
    elif changed == "top_fraction": plugin.detector.top_fraction = .01
    elif changed == "pixel_quantile": plugin.segmentation["pixel_quantile"] = .5
    elif changed == "rules": plugin.rules["zero_ng_quantile"] = .95
    else: write_image(ng, np.full((4, 4, 3), 80, np.uint8))
    plugin.prepare_category("cat", [ok], [ok])
    assert plugin.detector.calls > count
    assert plugin.thresholds["cat"]["ng_images"] == [str(ng)]
    assert plugin.thresholds["cat"]["calibration_ng"] == 1
    if changed == "pixel_quantile": assert plugin.thresholds["cat"]["pixel_threshold"] == 50


class Tensor:
    def __init__(self, value): self.value = value
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


def test_yolo_inference_clips_surviving_instances_to_roi(tmp_path):
    roi = tmp_path / "roi.png"
    selected = np.zeros((4, 4), bool); selected[:2, :2] = True
    write_image(roi, selected.astype(np.uint8) * 255)
    class Boxes:
        conf = Tensor(np.array([.8, .9]))
        xyxy = Tensor(np.array([[0, 0, 4, 4], [2, 2, 4, 4]]))
        def __len__(self): return 2
    masks = np.ones((2, 4, 4)); masks[1, :2] = 0
    result = SimpleNamespace(boxes=Boxes(), masks=SimpleNamespace(data=Tensor(masks)))
    detector = YoloSegDetector.__new__(YoloSegDetector)
    detector.model = SimpleNamespace(predict=lambda **kwargs: [result])
    detector.roi_mask = roi; detector._roi_cache = {}; detector.imgsz = 4
    detector.conf_floor = .001; detector.nms_iou = .7; detector.max_det = 100; detector.device = "cpu"
    score, confs, boxes, clipped = detector.infer(np.zeros((4, 4, 3), np.uint8))
    assert score == .8 and len(confs) == 1 and np.array_equal(clipped[0], selected)


def test_ablation_auroc_ties_in_calibration_and_evaluation(tmp_path, monkeypatch):
    ablation = cli("yolo_seg_ng_count_ablation")
    root = tmp_path / "category"; paths = []
    for label in ("NG", "OK"):
        path = root / label / f"{label}.png"; write_image(path, np.full((4, 4, 3), 100, np.uint8)); paths.append(str(path))
    write_image(root / "mask" / "NG_t.png", np.zeros((4, 4), np.uint8))
    def infer(*args):
        return .5, np.array([.5]), np.array([[0, 0, 4, 4]]), np.ones((1, 4, 4), bool), 1
    monkeypatch.setattr(ablation, "infer", infer)
    args = SimpleNamespace(max_fpr=1, target_recall=1, conf_floor=.001)
    thresholds = ablation.calibrate(None, root, paths, tmp_path / "out", args)
    result = ablation.evaluate(None, root, paths, tmp_path / "out", thresholds, args)
    assert thresholds["classification"]["auroc"] == result["test_auroc"] == .5
    assert result["iou_micro"] == 1


def test_lifecycle_reports_keep_hidden_truth_out_of_verified_metrics(tmp_path, monkeypatch):
    from detected_pipeline.config import load_project_config
    from detected_pipeline.contracts import Decision
    from fakes import FakePretrainedPlugin, make_image
    lifecycle = cli("run_lifecycle")
    category = "qiumian_fupai"
    config = load_project_config(Path(__file__).resolve().parents[1])
    config["categories"] = {category: {}}; config["workspace_root"] = str(tmp_path / "workspace")
    source = tmp_path / category
    make_image(source / "NG" / "caught.png", 50); make_image(source / "NG" / "miss.png", 80)
    for name in ("caught", "miss"):
        write_image(source / "mask" / f"{name}_t.png", np.zeros((12, 16), np.uint8))
    seed = make_image(tmp_path / "seed.png", 20)
    class FakeLifecycle:
        def __init__(self):
            self.plugin = FakePretrainedPlugin(tmp_path / "predictions", {"caught": (Decision.NG, False, False)})
            self.thresholds = {category: {"image_threshold": .5, "pixel_threshold": .5}}
        def prepare_category(self, *args): pass
        def calibrate(self, category, ng): return {"rule": "fake", "image_threshold": .5, "calibration_ng": len(ng)}
        def predict(self, *args): return self.plugin.predict(*args)
        def close(self): pass
    monkeypatch.setattr(lifecycle, "load_project_config", lambda _: config)
    monkeypatch.setattr(lifecycle, "initialization", lambda *args: ([], [seed], [], [], "stratified"))
    monkeypatch.setattr(lifecycle, "load_pretrained_plugin", lambda *args: FakeLifecycle())
    monkeypatch.setattr(lifecycle, "run_sampled_review", lambda *args: {"pseudo_ok": [0], "stats": {}})
    monkeypatch.setattr(sys, "argv", ["run_lifecycle.py", "--category", category, "--inbox", str(source / "NG"), "--initialization-manifest", "unused.json"])
    lifecycle.main()
    ws = tmp_path / "workspace"
    batch = json.loads((ws / "batch_reports" / category / "batch_0001.json").read_text())
    summary = cli("report_metrics").category_metrics(ws, category)
    assert batch["review_sampling"]["pseudo_ok_hidden_ng"] == 1
    assert batch["official"]["scope"] == summary["scope"] == "reviewed_online_subset"
    assert batch["official"]["count"] == summary["count"] == 1
    assert batch["official_segmentation"]["iou_micro"] == summary["iou_micro"] == 1
    assert summary["excluded_initialization"] == summary["excluded_unverified"] == 1
    with sqlite3.connect(ws / "state" / "pipeline.sqlite3") as db:
        assert db.execute("SELECT reviewed_decision FROM samples WHERE label_source='sampling_pseudo_ok'").fetchone()[0] == "OK"


def test_standalone_yolo_reports_same_micro_definition(tmp_path):
    standalone = cli("train_yolo")
    small = np.zeros((12, 12), bool); small[0, 0] = True
    large = np.zeros_like(small); large[:10, :10] = True
    items = []
    for name, gt in (("small", small), ("large", large)):
        item, _ = sample(tmp_path, name, gt)
        write_image(tmp_path / "masks" / f"{name}_t.png", (~gt).astype(np.uint8) * 255)
        items.append((Path(item["image"]), True))
    outputs = iter([(.9, small), (.1, large)])
    def infer(image):
        score, mask = next(outputs)
        return score, np.array([score]), np.array([[0, 0, 12, 12]]), mask[None]
    detector = SimpleNamespace(infer=infer, union=YoloSegDetector.union)
    metrics = standalone.evaluate(detector, items, {"image_threshold": .5, "mask_conf_threshold": .05},
                                  tmp_path / "test", None, tmp_path / "masks")
    assert metrics["iou_micro"] == pytest.approx(1 / 101)
    assert metrics["mean_iou_all_ng"] == metrics["recall"] == .5


def test_ablation_roi_exclusion_can_leave_no_evaluable_images(tmp_path, monkeypatch):
    ablation = cli("yolo_seg_ng_count_ablation")
    root = tmp_path / "category"; path = root / "NG" / "outside.png"
    write_image(path, np.full((4, 4, 3), 100, np.uint8))
    target = np.full((4, 4), 255, np.uint8); target[3, 3] = 0
    write_image(root / "mask" / "outside_t.png", target)
    roi = tmp_path / "roi.png"; selected = np.zeros((4, 4), np.uint8); selected[:2, :2] = 255
    write_image(roi, selected)
    monkeypatch.setitem(ablation.ROI_FILES, root.name, roi)
    monkeypatch.setattr(ablation, "infer", lambda *args: (0, np.zeros(0), np.zeros((0, 4)), np.zeros((0, 4, 4), bool), 1))
    thresholds = {"classification": {"threshold": .5, "auroc": .5, "recall": 1, "ok_false_positive_rate": 0,
                                     "max_fpr": .2, "fpr_cap_binding": False}, "segmentation": {"mask_conf_threshold": .5}}
    result = ablation.evaluate(None, root, [str(path)], tmp_path / "out", thresholds, SimpleNamespace())
    assert result["excluded_outside_roi"] == 1
    for name in ("iou_micro", "mean_iou_all_ng", "recall", "ok_false_positive_rate", "test_auroc", "p95_inference_ms"):
        assert result[name] is None
