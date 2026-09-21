from __future__ import annotations

import importlib
import os
from pathlib import Path


from detected_pipeline.config import load_yaml


class PluginLoadError(RuntimeError):
    pass


def load_pretrained_plugin(config_path: Path, project_root: Path | None = None):
    """Instantiate the pretrained detector from ``configs/pretrained.yaml`` (factory = module:function)."""
    if not config_path.is_file():
        raise PluginLoadError(f"pretrained plugin is not configured: {config_path}")
    spec = load_yaml(config_path)
    plugin_config = dict(spec.get("config", {}))
    plugin_config["project_root"] = str(project_root or config_path.resolve().parents[1])
    for key, env in (("cache_root", "PIPELINE_PRETRAINED_CACHE_ROOT"), ("results_root", "PIPELINE_PRETRAINED_RESULTS_ROOT")):
        if os.environ.get(env):
            plugin_config[key] = os.environ[env]
        elif not Path(plugin_config.get(key, "")).is_absolute():
            plugin_config[key] = str(Path(plugin_config["project_root"]) / plugin_config.get(key, key))
    factory_spec = spec.get("factory", "")
    if ":" not in factory_spec:
        raise PluginLoadError("plugin factory must use module:function syntax")
    module_name, factory_name = factory_spec.split(":", 1)
    try:
        plugin = getattr(importlib.import_module(module_name), factory_name)()
        plugin.load(plugin_config)
        health = plugin.healthcheck()
    except Exception as exc:
        raise PluginLoadError(f"failed to load pretrained plugin {factory_spec}: {exc}") from exc
    if not isinstance(health, dict) or not health.get("ok"):
        raise PluginLoadError(f"plugin healthcheck failed: {health!r}")
    return plugin
