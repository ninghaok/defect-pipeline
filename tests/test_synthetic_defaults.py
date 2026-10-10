"""Default augmentation is explicit: missing backend must not become a baseline run."""
import importlib.util
import sys
from pathlib import Path

import pytest

from detected_pipeline.config import load_project_config

PROJECT = Path(__file__).resolve().parents[1]


def test_enabled_default_and_explicit_baseline(monkeypatch):
    monkeypatch.delenv("PIPELINE_SYNTHETIC_CONFIG", raising=False)
    assert load_project_config(PROJECT)["synthetic"]["enabled"] is True
    monkeypatch.setenv("PIPELINE_SYNTHETIC_CONFIG", str(PROJECT / "configs/synthetic_baseline.yaml"))
    assert load_project_config(PROJECT)["synthetic"] == {"enabled": False}


def test_missing_policy_is_not_silently_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_SYNTHETIC_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        load_project_config(PROJECT)


def test_missing_backend_fails_before_workspace_or_inference(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("synthetic_default_lifecycle", PROJECT / "cli/run_lifecycle.py")
    lifecycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifecycle)
    monkeypatch.delenv("PIPELINE_SYNTHETIC_CONFIG", raising=False)
    monkeypatch.setenv("PIPELINE_RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.setattr(sys, "argv", ["run_lifecycle.py", "--category", "qiumian_fupai",
                                     "--inbox", str(tmp_path / "inbox"),
                                     "--initialization-manifest", str(tmp_path / "manifest.json")])
    with pytest.raises(ValueError, match="configure_synthetic.py"):
        lifecycle.main()
    assert not (tmp_path / "results").exists()
