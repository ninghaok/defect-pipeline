from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from detected_pipeline.util import atomic_copy, atomic_write_json, sha256_file, utc_now


class ModelRegistry:
    def __init__(self, workspace: Path):
        self.root = workspace / "model_registry"

    def register_candidate(
        self, category: str, model_version: str, checkpoint: Path,
        dataset_version: str, thresholds: dict[str, float], smoke_test: Callable[[Path], dict[str, Any]],
    ) -> dict[str, Any]:
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError(f"missing or empty checkpoint: {checkpoint}")
        version_root = self.root / category / "versions" / model_version
        stored = atomic_copy(checkpoint, version_root / "last.pt")
        digest = sha256_file(stored)
        try:
            smoke = smoke_test(stored)
            if not smoke.get("ok"):
                raise ValueError(f"smoke test failed: {smoke}")
            status = "candidate"
        except Exception as exc:
            metadata = {"model_version": model_version, "status": "failed", "error": str(exc), "sha256": digest}
            atomic_write_json(version_root / "model.json", metadata)
            raise
        metadata = {
            "model_version": model_version, "category": category, "status": status,
            "dataset_version": dataset_version, "checkpoint": str(stored), "sha256": digest,
            "thresholds": thresholds, "smoke_test": smoke, "created_at": utc_now(),
        }
        atomic_write_json(version_root / "model.json", metadata)
        return metadata

    def promote(self, category: str, metadata: dict[str, Any]) -> dict[str, Any]:
        category_root = self.root / category
        production_path = category_root / "production.json"
        previous = None
        if production_path.exists():
            previous = json.loads(production_path.read_text(encoding="utf-8"))
            atomic_write_json(category_root / "previous.json", previous)
            previous_meta = Path(previous["metadata_path"])
            if previous_meta.exists():
                archived = json.loads(previous_meta.read_text(encoding="utf-8"))
                archived["status"] = "archived"
                atomic_write_json(previous_meta, archived)
        metadata_path = category_root / "versions" / metadata["model_version"] / "model.json"
        production = {
            "category": category, "model_version": metadata["model_version"],
            "checkpoint": metadata["checkpoint"], "sha256": metadata["sha256"],
            "thresholds": metadata["thresholds"], "metadata_path": str(metadata_path),
            "promoted_at": utc_now(),
        }
        metadata = dict(metadata)
        metadata["status"] = "production"
        atomic_write_json(metadata_path, metadata)
        atomic_write_json(production_path, production)
        return production

    def rollback(self, category: str) -> dict[str, Any]:
        category_root = self.root / category
        previous_path = category_root / "previous.json"
        if not previous_path.exists():
            raise RuntimeError(f"no previous model for {category}")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        atomic_write_json(category_root / "production.json", previous)
        return previous

    def current(self, category: str) -> dict[str, Any] | None:
        path = self.root / category / "production.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def latest_candidate(self, category: str) -> dict[str, Any] | None:
        versions = self.root / category / "versions"
        candidates = []
        for path in versions.glob("*/model.json") if versions.exists() else []:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status") in {"shadow_candidate", "candidate_insufficient_or_rejected"}:
                candidates.append(data)
        return max(candidates, key=lambda item: item.get("created_at", "")) if candidates else None
