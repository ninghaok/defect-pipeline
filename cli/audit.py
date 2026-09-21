from __future__ import annotations

import json

from _common import context
from detected_pipeline.audit import audit_workspace


if __name__ == "__main__":
    _, workspace, categories, _ = context()
    report = audit_workspace(workspace, categories)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "ok" else 1)

