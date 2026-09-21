from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detected_pipeline.config import load_project_config  # noqa: E402


def context():
    config = load_project_config(PROJECT_ROOT)
    workspace = Path(config["workspace_root"])
    categories = list(config["categories"])
    return PROJECT_ROOT, workspace, categories, config

