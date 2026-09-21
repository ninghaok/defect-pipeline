from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return data


def load_project_config(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    configs = project_root / "configs"
    pipeline = load_yaml(configs / "pipeline.yaml")
    pipeline["categories"] = load_yaml(configs / "categories.yaml")["categories"]
    pipeline["training"] = load_yaml(configs / "training.yaml")
    pipeline["roi"] = load_yaml(configs / "roi.yaml").get("categories", {}) if (configs / "roi.yaml").is_file() else {}
    pipeline["project_root"] = str(project_root)
    pipeline["pretrained_config"] = str(configs / pipeline.get("pretrained_config", "pretrained.yaml"))
    override = os.environ.get("PIPELINE_RESULTS_ROOT")
    workspace = Path(override) / "workspace" if override else Path(pipeline["workspace_root"])
    pipeline["workspace_root"] = str(workspace if workspace.is_absolute() else project_root / workspace)
    if os.environ.get("PIPELINE_TRAINING_EPOCHS"):   # smoke tests only
        pipeline["training"]["epochs"] = int(os.environ["PIPELINE_TRAINING_EPOCHS"])
    checkpoint = Path(pipeline["training"]["base_checkpoint"])
    if not checkpoint.is_absolute():
        pipeline["training"]["base_checkpoint"] = str(project_root / checkpoint)
    return pipeline


def roi_mask_for(config: dict[str, Any], category: str) -> Path | None:
    entry = config.get("roi", {}).get(category)
    if not entry:
        return None
    mask = Path(entry["mask"])
    return mask if mask.is_absolute() else Path(config["project_root"]) / mask
