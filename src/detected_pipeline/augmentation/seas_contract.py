"""Audited SeaS request/provenance contract, independent of model dependencies."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


TRUSTED = {"folder_ground_truth": "explicit_ground_truth_review", "human_review": "human_confirmed",
           "initial_calibration_yolo_train_ok": "trusted_initialization_ok"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class ReloadablePretrained:
    """Stable lifecycle object identity, with explicit GPU-release/reload support."""
    def __init__(self, factory):
        self._factory = factory
        self._delegate = factory()
        self._preparations = []

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def prepare_category(self, category, train_ok, calibration_ok):
        result = self._delegate.prepare_category(category, train_ok, calibration_ok)
        self._preparations = [r for r in self._preparations if r[0] != category]
        self._preparations.append((category, list(train_ok), list(calibration_ok)))
        return result

    def close(self):
        if self._delegate is not None:
            self._delegate.close()

    @contextmanager
    def release_for_generation(self):
        import numpy as np
        import torch
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        thresholds = copy.deepcopy(self._delegate.thresholds)
        self._delegate.close()
        self._delegate = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            yield
        finally:
            try:
                self._delegate = self._factory()
                for category, train_ok, calibration_ok in self._preparations:
                    self._delegate.prepare_category(category, train_ok, calibration_ok)
                if identity(self._delegate.thresholds) != identity(thresholds):
                    raise RuntimeError("Pretrained reload changed calibrated thresholds")
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.set_rng_state(torch_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)


def observation_index(workspace, category):
    """Use only already-written batch reports, never hidden simulation truth."""
    observations = {}
    position = batch = 0
    for path in sorted((Path(workspace) / "batch_reports" / category).glob("batch_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        batch = int(report["batch"])
        for row in report["rows"]:
            position += 1
            if row.get("review") == "reviewed":
                observations[row["sample_id"]] = {
                    "confirmed_review_batch": batch, "observed_stream_position": position,
                    "reviewed_label": row["truth"], "label_source": row["label_source"],
                }
    return observations, batch, position


def feedback_identity(workspace):
    """Verify the generator did not mutate real feedback/pool membership."""
    with sqlite3.connect(Path(workspace) / "state/pipeline.sqlite3") as db:
        rows = {table: db.execute("SELECT * FROM " + table + " ORDER BY 1,2").fetchall()
                for table in ("samples", "pool_membership", "dataset_split_assignment")}
    return identity(rows)


def support_record(row, label, observations, initial_bank_sha):
    source = row.get("label_source")
    if source not in TRUSTED:
        raise ValueError("Generator support is not a real reviewed label: " + str(source))
    if source == "initial_calibration_yolo_train_ok":
        if label != "OK" or row["sha256"] not in initial_bank_sha or row.get("camera_id") != "initialization":
            raise ValueError("Invalid initialization support provenance")
        observed = {"confirmed_review_batch": 0, "observed_stream_position": 0}
    else:
        observed = observations.get(row["sample_id"])
        if not observed or observed["reviewed_label"] != label or observed["label_source"] != source:
            raise ValueError("Support does not have a matching already-reviewed batch record")
    image = Path(row["copy_path"]).resolve(strict=True)
    if digest(image) != row["sha256"]:
        raise ValueError("Real support image changed")
    result = {"sample_id": row["sample_id"], "image": str(image), "image_sha256": row["sha256"],
              "origin": "real", "split": "train", "label": label, "label_source": TRUSTED[source],
              "confirmed_review_batch": observed["confirmed_review_batch"],
              "observed_stream_position": observed["observed_stream_position"]}
    if label == "NG":
        mask = Path(row["mask"]).resolve(strict=True)
        result.update(mask=str(mask), mask_sha256=digest(mask), mask_convention="white_defect")
    return result


def validate_generated(manifest_path, request_path, forbidden_sha, allow_unreviewed):
    """Hash and provenance gate, never describe predicted masks as human ground truth."""
    import numpy as np
    from PIL import Image
    manifest_path = Path(manifest_path).resolve(strict=True)
    root = manifest_path.parent
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("request_sha256") != digest(request_path):
        raise ValueError("Generation manifest status/request mismatch")
    chosen, seen = [], set()
    for row in result.get("items", []):
        selection = row.get("selection_status")
        human = row.get("approved_for_training") is True and selection == "human_approved"
        if not human and not (allow_unreviewed and selection == "structurally_valid_unreviewed"):
            continue
        paths = []
        for kind in ("image", "mask"):
            path = Path(row[kind])
            path = (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
            if root not in path.parents or digest(path) != row[kind + "_sha256"]:
                raise ValueError("Generated file escaped event directory or failed checksum")
            paths.append(path)
        if row["image_sha256"] in forbidden_sha or row["image_sha256"] in seen:
            raise ValueError("Synthetic image duplicates real/held-out data or another generated image")
        seen.add(row["image_sha256"])
        with Image.open(paths[0]) as image, Image.open(paths[1]) as mask:
            if image.size != mask.size or mask.mode not in ("L", "1"):
                raise ValueError("Synthetic mask/image dimensions or mask mode invalid")
            raw = np.asarray(mask)
            if raw.dtype == np.bool_:
                raw = raw.astype(np.uint8) * 255
            if not np.isin(raw, [0, 255]).all() or not (raw > 0).any() or (raw > 0).all():
                raise ValueError("Synthetic mask must be nonempty/nonfull binary white-defect")
        chosen.append({**row, "image": str(paths[0]), "mask": str(paths[1]),
                       "human_approved": human})
    return result, chosen
