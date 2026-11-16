"""CPU lifecycle integration: real gate/registry/state, deterministic fake inference/training."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from detected_pipeline.config import load_project_config
from detected_pipeline.contracts import Decision
from detected_pipeline.promotion import PROMOTION_RULE
from detected_pipeline.registry import ModelRegistry
from detected_pipeline.roi import write_image
from fakes import FakePretrainedPlugin, make_image


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("initial_model", ["pretrained", "yolo"])
@pytest.mark.parametrize("shadow_fraction,legacy,extra_fp,expected", [
    (1.0, False, False, "production"), (.25, False, False, "rejected_shadow"),
    (1.0, True, False, "rejected_shadow"), (1.0, False, True, "production")])
def test_shadow_only_promotion_and_resume(tmp_path, monkeypatch, initial_model, shadow_fraction, legacy, extra_fp, expected):
    root = Path(__file__).resolve().parents[1]
    lifecycle = load_module("promotion_lifecycle_test", root / "cli/run_lifecycle.py")
    # Exercise the real CPU segmentation function without loading GPU feature networks.
    segmentation = load_module("promotion_segmentation_test", root / "src/detected_pipeline/pretrained/segmentation.py")
    monkeypatch.setitem(sys.modules, "detected_pipeline.pretrained", SimpleNamespace(hybrid_regions=segmentation.hybrid_regions))
    category = "qiumian_fupai"
    config = load_project_config(root)
    config.update(categories={category: {}}, workspace_root=str(tmp_path / "ws"), review_batch_size=2,
                  review_sampling={"enabled": False})
    config["synthetic"] = {"enabled": False}  # Gate tests do not run an external generator.
    config["lifecycle"].update(first_train_ng=1, retrain_increment=100, shadow_min_ng=2)
    source = tmp_path / category; paths = []
    for i in range(1, 4):
        paths += [make_image(source / "NG" / f"ng{i}.png", 80 + i), make_image(source / "OK" / f"ok{i}.png", 20 + i)]
        write_image(source / "mask" / f"ng{i}_t.png", np.zeros((12, 16), np.uint8))
    cal_ok = make_image(tmp_path / "cal_ok.png", 10)

    class Adapter(FakePretrainedPlugin):
        fraction = shadow_fraction
        alarm_ok = False
        def predict(self, image, context):
            self.decisions[image.stem] = (Decision.NG if image.stem.startswith("ng") or self.alarm_ok else Decision.OK, False, False)
            result = super().predict(image, context)
            mask = np.zeros((12, 16), np.uint8)
            if result.final_decision == Decision.NG: mask[:, :int(16 * self.fraction)] = 255
            write_image(Path(result.binary_mask_path), mask)
            return result

    class Pretrained(Adapter):
        fraction = .25 if extra_fp else .5
        def __init__(self):
            super().__init__(tmp_path / "pretrained")
            self.thresholds = {category: {"image_threshold": .5, "pixel_threshold": .5}}
            self.segmentation = {"min_area": 1}
            self.detector = self
        def prepare_category(self, *args): pass
        def calibrate(self, category, ng): return dict(rule="fake", image_threshold=.5, calibration_ng=len(ng))
        def heatmap(self, category, image):
            heat = np.zeros(image.shape[:2], np.float32)
            if image.max() > 50: heat[:, :8] = 1
            return heat, np.ones_like(heat, bool)
        def image_score(self, heat, valid): return float(heat.max())

    def train(workspace, category, milestone, *args):
        row = lifecycle.confirmed_rows(workspace, category, "NG")[0]
        checkpoint = tmp_path / "fake.pt"; checkpoint.write_bytes(b"fake model")
        return dict(checkpoint=str(checkpoint), model_version="test-seg-v1", thresholds=dict(image_threshold=.5, mask_conf_threshold=.5),
                    calibration={"error_cost": 999}, counts={}, calibration_records=[dict(image=row["copy_path"], label="NG", score=.9),
                                                                    dict(image=str(cal_ok), label="OK", score=0)])

    monkeypatch.setattr(lifecycle, "load_project_config", lambda _: config)
    monkeypatch.setattr(lifecycle, "initialization", lambda *args: ([], [], [cal_ok], [], "random"))
    monkeypatch.setattr(lifecycle, "random_stream", lambda found, _: [p for p in paths if p in found])
    monkeypatch.setattr(lifecycle, "load_pretrained_plugin", lambda *args: Pretrained())
    monkeypatch.setattr(lifecycle, "YoloSegDetector", lambda *args: pytest.fail("No offline comparison inference"))
    def adapter(model, ws, cfg, tag, roi):
        result = Adapter(tmp_path / tag)
        if model["model_version"] == "old": result.fraction = .25 if extra_fp else .5
        else: result.alarm_ok = extra_fp
        return result
    monkeypatch.setattr(lifecycle, "yolo_adapter", adapter)
    monkeypatch.setattr(lifecycle, "train_candidate", train)
    monkeypatch.setattr(lifecycle, "smoke_test_seg", lambda *args: {"ok": True})
    monkeypatch.setattr(lifecycle, "batch_snapshot", lambda *args: {"lifecycle_complete": True})
    monkeypatch.setattr(lifecycle, "fixed_test_metrics", lambda *args: pytest.fail("Test metrics must not enter promotion"))
    monkeypatch.setattr(sys, "argv", ["run_lifecycle.py", "--category", category, "--inbox", str(source),
                                     "--initialization-manifest", "unused.json"])
    # First run ends after one shadow batch; no automatic promotion at end-of-stream.
    withheld = [p.read_bytes() for p in paths[-2:]]
    for p in paths[-2:]: p.unlink()
    if initial_model == "yolo":
        checkpoint = tmp_path / "old.pt"; checkpoint.write_bytes(b"old model")
        registry = ModelRegistry(Path(config["workspace_root"]))
        old = registry.register_candidate(category, "old", checkpoint, "old", dict(image_threshold=.5, mask_conf_threshold=.5), lambda _: {"ok": True})
        registry.promote(category, old)
    lifecycle.main()
    ws = Path(config["workspace_root"])
    state_path = ws / "state" / f"lifecycle_{category}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    model_path = ws / "model_registry" / category / "versions/test-seg-v1/model.json"
    metadata = json.loads(model_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "shadow_candidate"
    assert metadata["calibration"]["error_cost"] == 999  # calibration quality never vetoes entry
    assert "offline_comparison" not in metadata
    assert not list((ws / "promotion_reports").rglob("offline*.json"))
    assert state["shadow_batches"] == 1 and state["candidate"] is not None
    if legacy:
        for pair in state["shadow_rows"]:
            for role in ("official_metrics", "shadow_metrics"):
                pair[role]["segmentation_metric_version"] = "obsolete"
        state_path.write_text(json.dumps(state), encoding="utf-8")
    # Mutating historical output files must not alter persisted first-batch pixel counts.
    for p in (tmp_path / "shadow_yolo").glob("*.png"): write_image(p, np.zeros((12, 16), np.uint8))
    for p, data in zip(paths[-2:], withheld): p.write_bytes(data)
    lifecycle.main()
    metadata = json.loads(model_path.read_text(encoding="utf-8"))
    assert metadata["status"] == expected
    comparison = metadata["shadow_comparison"]
    if expected == "rejected_shadow":
        if legacy: assert comparison["reason"] == "missing_or_legacy_tolerance_counts"
        else: assert comparison["checks"]["tolerance_improved"] is False
        current = ModelRegistry(ws).current(category)
        assert (current["model_version"] if current else None) == ("old" if initial_model == "yolo" else None)
        return
    assert comparison["rule"] == PROMOTION_RULE
    assert comparison["batches"] == 2 and comparison["ng"] == 2
    assert comparison["official"]["iou_micro"] == (.25 if extra_fp else .5)
    assert comparison["candidate"]["iou_micro"] == 1
    assert comparison["official"]["fn"] == comparison["candidate"]["fn"] == 0
    assert comparison["official"]["fp"] == 0
    assert comparison["candidate"]["fp"] == (2 if extra_fp else 0)
    if extra_fp:
        assert comparison["reason"] == "tolerance_gain_at_least_20_points"
    assert comparison["improved_metrics"] == ["tolerant_agreement_micro"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["candidate"] is None
