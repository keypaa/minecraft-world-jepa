"""Configuration loading helpers."""

from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load a stage config and merge its optional base config."""
    path = Path(config_path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    base_name = cfg.pop("inherits", None)
    if not base_name:
        return cfg

    base_path = path.with_name(f"{base_name}.yaml")
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    merged = dict(base_cfg)
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged
