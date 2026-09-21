from __future__ import annotations

import json
from pathlib import Path

from detected_pipeline.feedback import FeedbackStore
from detected_pipeline.util import atomic_write_json, sha256_file, utc_now


def audit_workspace(workspace: Path, categories: list[str]) -> dict:
    store = FeedbackStore(workspace, categories)
    errors = []
    category_rows = {}
    for category in categories:
        counts = store.counts(category)
        state = store.state(category)
        production = workspace / "model_registry" / category / "production.json"
        model = json.loads(production.read_text(encoding="utf-8")) if production.exists() else None
        if model:
            checkpoint = Path(model["checkpoint"])
            if not checkpoint.exists() or sha256_file(checkpoint) != model["sha256"]:
                errors.append(f"{category}: production checkpoint integrity failure")
        category_rows[category] = {"counts": counts, "state": state, "production": model}
    report = {"status": "ok" if not errors else "error", "created_at": utc_now(), "errors": errors, "categories": category_rows}
    atomic_write_json(workspace / "reports" / "audit.json", report)
    return report

