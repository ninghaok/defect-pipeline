"""Read-only batch-end observations. No evaluation truth enters model decisions."""
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from detected_pipeline.masks import internal_mask
from detected_pipeline.training.seg_lifecycle import confirmed_rows


def batch_snapshot(workspace, category, state, config, calibration_sha, roi_mask=None):
    life = config["lifecycle"]
    ok = [r for r in confirmed_rows(workspace, category, "OK", life.get("pseudo_ok_in_training", True))
          if r["sha256"] not in calibration_sha]
    ng = confirmed_rows(workspace, category, "NG")
    with closing(sqlite3.connect(workspace / "state" / "pipeline.sqlite3")) as db:
        assigned = dict(db.execute("SELECT sample_id,split FROM dataset_split_assignment WHERE category=?", (category,)))
    train_ng = excluded_roi = 0
    for row in ng:
        mask = internal_mask(row["mask"])
        if roi_mask:
            mask &= internal_mask(roi_mask, mask.shape)
        if not mask.any():
            excluded_roi += 1
            continue
        split = assigned.get(row["sample_id"])
        if split is None:
            token = f"{life.get('split_seed', 42)}:{category}:{row['sha256']}"
            value = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16) / float(16 ** 16)
            split = "calibration" if value < life.get("ng_calibration_fraction", .25) else "train"
        train_ng += split == "train"
    latest = next((e for e in reversed(state["history"]) if e["event"] == "candidate_trained"), None)
    latest_model = None
    if latest:
        path = workspace / "model_registry" / category / "versions" / latest["model"] / "model.json"
        model = json.loads(path.read_text(encoding="utf-8"))
        latest_model = {k: model.get(k) for k in ("model_version", "status", "milestone", "fixed_test")}
    return {"lifecycle_complete": True, "batch": state["reviewed_batches"], "last_milestone": state["last_milestone"],
            "eligible_training_ok": len(ok), "eligible_training_ng": train_ng,
            "eligible_pseudo_ok": sum(r["label_source"] == "sampling_pseudo_ok" for r in ok),
            "confirmed_valid_ng": len(ng), "excluded_training_ng_outside_roi": excluded_roi,
            "latest_yolo": latest_model}
