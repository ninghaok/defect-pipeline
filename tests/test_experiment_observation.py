import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from detected_pipeline.util import atomic_write_json
from detected_pipeline.roi import write_image


def test_full_stream_export_counts_hidden_miss_without_mutating_batch(tmp_path):
    cli = Path(__file__).resolve().parents[1] / "cli"
    sys.path.insert(0, str(cli))
    spec = importlib.util.spec_from_file_location("export_simulation_table", cli / "export_simulation_table.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ws = tmp_path / "workspace"; rows = []
    for name, decision, size in (("caught", "NG", 1), ("miss", "OK", 100)):
        image = tmp_path / "stream" / "NG" / f"{name}.png"
        gt = np.full((12, 12), 255, np.uint8); gt.flat[:size] = 0
        mask = tmp_path / "predictions" / f"{name}.png"
        write_image(image, np.full((12, 12, 3), 80, np.uint8))
        write_image(image.parent.parent / "mask" / f"{name}_t.png", gt)
        write_image(mask, (gt < 128).astype(np.uint8) * 255)
        rows.append({"image": str(image), "official": decision, "official_mask": str(mask),
                     "truth": "NG" if decision == "NG" else None, "hidden_truth": "NG"})
    report = {"batch": 1, "rows": rows, "reviewed": 1, "official_model": "pretrained", "official": {"recall": 1},
              "end_of_batch": {"lifecycle_complete": True, "latest_yolo": None, "eligible_training_ok": 201, "eligible_training_ng": 1}}
    path = ws / "batch_reports" / "cat" / "batch_0001.json"; atomic_write_json(path, report)
    before = path.read_bytes()
    result = module.export_category(ws, "cat", {}, tmp_path / "summary")
    saved = json.loads((tmp_path / "summary" / "cat_table.json").read_text(encoding="utf-8"))
    assert result["completed_rows"] == 1 and len(saved["headers"]) == 15
    row = saved["rows"][0]
    assert row["在线Recall"] == .5 and row["在线micro IoU"] == 1 / 101
    assert row["实际的输入OK/NG数"] == "0/2" and row["需要人工打标的数量"] == 1
    assert path.read_bytes() == before


def test_batch_snapshot_does_not_assign_new_training_splits(tmp_path):
    import sqlite3
    from detected_pipeline.feedback import FeedbackStore
    from detected_pipeline.experiment_observation import batch_snapshot
    store = FeedbackStore(tmp_path / "workspace", ["cat"])
    image = tmp_path / "ok.png"; write_image(image, np.full((4, 4, 3), 80, np.uint8))
    store.seed_confirmed_ok_for_training(image, "cat", "seed")
    config = {"lifecycle": {"split_seed": 42, "ng_calibration_fraction": .25}}
    state = {"history": [], "reviewed_batches": 1, "last_milestone": 0}
    with sqlite3.connect(store.db_path) as db: before = db.execute("SELECT * FROM dataset_split_assignment").fetchall()
    result = batch_snapshot(tmp_path / "workspace", "cat", state, config, set())
    with sqlite3.connect(store.db_path) as db: after = db.execute("SELECT * FROM dataset_split_assignment").fetchall()
    assert result["eligible_training_ok"] == 1 and result["eligible_training_ng"] == 0
    assert before == after
