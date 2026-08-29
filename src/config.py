"""Configuration loading.

The active config is selected with the SUBOCEANNET_CONFIG environment variable
(path to a YAML file); it defaults to <project root>/configs/config.yaml.
Config is re-read per call so tests can point it at a temporary environment.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ENV_VAR = "SUBOCEANNET_CONFIG"


def config_path() -> Path:
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override)
    return PROJECT_ROOT / "configs" / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load the YAML config, resolving relative paths against the project root."""
    path = config_path()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    def _resolve(value):
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, str) and "/" in value and not Path(value).is_absolute():
            cand = (PROJECT_ROOT / value).resolve()
            if cand.exists() or value.startswith("data"):
                return str(cand)
            return str(cand)
        return value

    cfg = _resolve(cfg)
    cfg["_path"] = str(path)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


@lru_cache(maxsize=1)
def _cached() -> Dict[str, Any]:  # pragma: no cover - convenience helper
    return load_config()


def get_config() -> Dict[str, Any]:
    """Return the current config dict (fresh read; cheap enough per request)."""
    return load_config()


def input_variable_names(cfg: Dict[str, Any]) -> list[str]:
    return [v["name"] for v in cfg["input_variables"]]


def depths(cfg: Dict[str, Any]) -> list[int]:
    return list(cfg["depths_m"])


def bundle_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["checkpoints_dir"]) / cfg["paths"]["bundle_name"]
