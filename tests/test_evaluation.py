import json
import tempfile
from pathlib import Path

import numpy as np

from detected_pipeline.evaluation import fixed_test_metrics, score_fixed_test, write_test_report
from detected_pipeline.roi import write_image


def test_fixed_test_cache_metrics_and_case_folders():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder); image = np.full((32, 32, 3), 90, np.uint8)
        for name in ("ok1", "ok2", "ng1", "ng2"):
            write_image(root / f"{name}.png", image)
        gt = np.full((32, 32), 255, np.uint8); gt[8:16, 8:16] = 0; write_image(root / "ng_mask.png", gt)
        items = [{"image": str(root / "ok1.png"), "label": "OK", "mask": None}, {"image": str(root / "ok2.png"), "label": "OK", "mask": None},
                 {"image": str(root / "ng1.png"), "label": "NG", "mask": str(root / "ng_mask.png")},
                 {"image": str(root / "ng2.png"), "label": "NG", "mask": str(root / "ng_mask.png")}]
        scores = {"ok1": .1, "ok2": .4, "ng1": .9, "ng2": .3}; calls = []

        def predict(path):
            calls.append(path.stem); mask = np.zeros((32, 32), bool); mask[8:16, 8:12] = True
            heat = np.random.default_rng(0).random((32, 32)).astype(np.float32)
            return {"score": scores[path.stem], "mask": mask, "heat": heat, "boxes": [[8, 8, 12, 16, "r1"]], "extra": {"note": path.stem}}
        cache = root / "cache" / "model-a"
        rows = score_fixed_test(items, predict, cache, "model-a")
        assert score_fixed_test(items, None, cache, "model-a") == rows and len(calls) == 4
        assert rows[2]["iou"] == 0.5 and (Path(rows[2]["visuals_dir"]) / "boxed.jpg").is_file()
        strict = fixed_test_metrics(rows, .5); loose = fixed_test_metrics(rows, .2)
        assert strict["recall"] == .5 and strict["ok_false_positive_rate"] == 0.0 and strict["mean_iou_all_ng"] == .25
        assert loose["recall"] == 1.0 and loose["ok_false_positive_rate"] == .5 and loose["test_auroc"] == .75
        workspace = root / "ws"
        out = write_test_report(workspace, "cat", "model-a", "candidate", rows, .5, cache)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert report["counts_by_case"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 2}
        fn_dir = out / "fn" / "00003"
        assert (fn_dir / "original.png").is_file() and (fn_dir / "original_mask.png").is_file()
        assert (fn_dir / "pred_mask.png").is_file() and (fn_dir / "heatmap.jpg").is_file() and (fn_dir / "boxed.jpg").is_file()
        assert json.loads((fn_dir / "score.json").read_text())["note"] == "ng2"
        assert (workspace / "test_reports" / "cat" / "test_history.jsonl").read_text().count("\n") == 1
        assert not (out / "tn" / "00000" / "original_mask.png").exists()
