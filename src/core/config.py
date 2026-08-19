from __future__ import annotations

"""
Configuration loader.

Loads YAML config files and merges them (default + environment overrides).
"""

import os
from pathlib import Path
from typing import Any

import yaml


_config: dict | None = None


def load_config(
    config_dir: str | Path = "config",
    environment: str | None = None,
) -> dict:
    """
    Load configuration from YAML files.

    Loads default.yaml first, then merges environment-specific overrides.

    Args:
        config_dir: Path to config directory
        environment: Optional environment name (e.g., "development", "production")

    Returns:
        Merged configuration dictionary
    """
    global _config

    config_dir = Path(config_dir)
    default_path = config_dir / "default.yaml"

    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    # Load default config
    with open(default_path) as f:
        config = yaml.safe_load(f) or {}

    # Merge environment overrides
    if environment is None:
        environment = os.environ.get("INDIRA_ENV", None)

    if environment:
        env_path = config_dir / f"{environment}.yaml"
        if env_path.exists():
            with open(env_path) as f:
                env_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, env_config)

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    _config = config
    return config


def get_config() -> dict:
    """Get the current loaded configuration."""
    if _config is None:
        return load_config()
    return _config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides.

    Format: INDIRA__{SECTION}__{KEY}=value
    Example: INDIRA__LLM__MODEL=qwen2.5:32b
    """
    prefix = "INDIRA__"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            parts = key[len(prefix) :].lower().split("__")
            _set_nested(config, parts, value)
    return config


def _set_nested(d: dict, keys: list[str], value: str) -> None:
    """Set a nested dictionary value from a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    # Try to parse as YAML value (for booleans, numbers, etc.)
    try:
        d[keys[-1]] = yaml.safe_load(value)
    except yaml.YAMLError:
        d[keys[-1]] = value
