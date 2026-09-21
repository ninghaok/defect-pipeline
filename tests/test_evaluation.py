import tempfile
from pathlib import Path

import numpy as np

from detected_pipeline.evaluation import fixed_test_metrics, score_fixed_test
from detected_pipeline.roi import write_image


def test_fixed_test_scores_are_cached_and_metrics_follow_the_threshold():
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
            return scores[path.stem], mask
        cache = root / "cache.json"
        rows = score_fixed_test(items, predict, cache, "model-a")
        rows_again = score_fixed_test(items, predict, cache, "model-a")
        assert len(calls) == 4 and rows == rows_again
        assert rows[2]["iou"] == 0.5
        strict = fixed_test_metrics(rows, .5); loose = fixed_test_metrics(rows, .2)
        assert strict["recall"] == .5 and strict["ok_false_positive_rate"] == 0.0 and strict["mean_iou_all_ng"] == .25
        assert loose["recall"] == 1.0 and loose["ok_false_positive_rate"] == .5 and loose["mean_iou_all_ng"] == .5
        assert loose["test_auroc"] == .75
        score_fixed_test(items, predict, cache, "model-b")
        assert len(calls) == 8   # a different model key re-scores
