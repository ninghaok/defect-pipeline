import tempfile
import unittest
from pathlib import Path

import numpy as np

from detected_pipeline.feedback import FeedbackStore
from detected_pipeline.roi import write_image
from detected_pipeline.training.seg_lifecycle import (assign_ng_splits, compare_models, materialize, next_milestone,
                                                     polygons, select_training_ok)

CATEGORIES = ["qiumian_fupai", "qiumian_xiepai", "di_mian_detection", "wa_yuan_detection"]


class LifecycleTests(unittest.TestCase):
    def test_milestone_schedule(self):
        life = {"first_train_ng": 40, "retrain_increment": 20, "retrain_increment_after": 100, "retrain_increment_late": 40}
        self.assertIsNone(next_milestone(39, 0, life))
        self.assertEqual(next_milestone(40, 0, life), 40)
        self.assertEqual(next_milestone(75, 40, life), 60)
        self.assertIsNone(next_milestone(75, 60, life))
        self.assertEqual(next_milestone(140, 100, life), 140)
        self.assertIsNone(next_milestone(139, 100, life))

    def test_persistent_ng_split_is_stable_and_roughly_25_percent(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder); FeedbackStore(workspace, CATEGORIES)
            rows = [{"sample_id": f"s{i}", "sha256": f"{i:064x}"} for i in range(400)]
            import sqlite3
            from contextlib import closing
            with closing(sqlite3.connect(workspace / "state" / "pipeline.sqlite3")) as db, db:
                for r in rows:
                    db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)", (r["sample_id"], r["sha256"], CATEGORIES[0], "x", "b", "c", "NG", "NG", "folder_ground_truth", None, "t"))
            train, cal = assign_ng_splits(workspace, CATEGORIES[0], rows, 0.25, 42)
            self.assertEqual(len(train) + len(cal), 400); self.assertTrue(70 <= len(cal) <= 130)
            train2, cal2 = assign_ng_splits(workspace, CATEGORIES[0], rows[::-1], 0.5, 7)   # roles never change
            self.assertEqual({r["sample_id"] for r in cal2}, {r["sample_id"] for r in cal})

    def test_training_ok_is_batch_stratified_and_excludes_calibration(self):
        rows = [{"sample_id": f"{b}{i}", "sha256": f"{b}{i}", "batch_id": b} for b in "ab" for i in range(10)]
        picked = select_training_ok(rows, {"a0", "a1"}, 6, 42)
        self.assertEqual(len(picked), 6); self.assertEqual(sum(r["batch_id"] == "a" for r in picked), 3)
        self.assertFalse({"a0", "a1"} & {r["sha256"] for r in picked})

    def test_polygons_and_materialize_with_roi_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); image = np.full((64, 64, 3), 90, np.uint8); write_image(root / "ng.png", image); write_image(root / "ok.png", image)
            mask = np.zeros((64, 64), np.uint8); mask[10:20, 10:20] = 255; write_image(root / "ng_mask.png", mask)
            outside = np.zeros((64, 64), np.uint8); outside[50:60, 50:60] = 255; write_image(root / "out_mask.png", outside)
            roi = np.zeros((64, 64), np.uint8); roi[:32] = 255; write_image(root / "roi.png", roi)
            self.assertEqual(len(polygons(mask >= 128)), 1)
            yaml_path, stats = materialize(root / "ds", [(root / "ng.png", root / "ng_mask.png"), (root / "ok.png", None), (root / "ng.png", root / "out_mask.png")], [(root / "ok.png", None)], root / "roi.png")
            self.assertTrue(yaml_path.is_file()); self.assertEqual(stats["train"]["images"], 2); self.assertEqual(len(stats["excluded"]), 1)
            self.assertEqual((root / "ds" / "labels" / "train" / "00001.txt").read_text(), "")

    def test_offline_gate_requires_no_extra_misses_and_a_real_gain(self):
        labels = [True] * 4 + [False] * 10
        official = [.9, .9, .9, .1] + [.05] * 8 + [.5, .5]
        candidate = [.9, .9, .9, .9] + [.05] * 8 + [.5, .5]
        gate = {"max_fpr_increase": 0.0, "min_fpr_reduction_for_equal_fn": 0.01}
        self.assertEqual(compare_models(official, .4, candidate, .4, labels, gate)["decision"], "promote")
        self.assertEqual(compare_models(official, .4, official, .4, labels, gate)["decision"], "reject")
        worse = [.9, .9, .9, .9] + [.6] * 3 + [.05] * 5 + [.5, .5]
        self.assertEqual(compare_models(official, .4, worse, .4, labels, gate)["decision"], "reject")


if __name__ == "__main__":
    unittest.main()
