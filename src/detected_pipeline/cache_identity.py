"""Content identities for derived artifacts. Threshold-only reporting is not inference."""
import hashlib
import json
from pathlib import Path

from detected_pipeline.util import sha256_file


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_identity(path):
    if not path:
        return None
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path) if path.is_file() else None}


def test_identity(items, model_key, roi_path, inference_settings=None):
    return fingerprint({"schema": 2, "model": model_key, "roi": file_identity(roi_path),
                        "inference": inference_settings or {},
                        "items": [{"image": file_identity(i["image"]), "label": i["label"],
                                   "mask": file_identity(i.get("mask"))} for i in items]})
