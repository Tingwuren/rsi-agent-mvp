"""Configuration loading, environment variable interpolation, and storage guards."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from rsi_mvp.schemas import ExperimentConfig

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class StorageGuardError(RuntimeError):
    """Raised when a configured artifact/cache directory lacks safe free disk space."""


def load_dotenv_file(dotenv_path: Path, override: bool = False) -> dict[str, str]:
    """Load simple KEY=VALUE entries from a ``.env`` file into ``os.environ``."""
    loaded: dict[str, str] = {}
    if not dotenv_path.is_file():
        return loaded

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _interpolate_env(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` inside parsed YAML."""
    if isinstance(obj, str):
        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default_val = match.group(2)
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default_val is not None:
                return default_val
            return match.group(0)

        return _ENV_PATTERN.sub(_replace, obj)
    if isinstance(obj, list):
        return [_interpolate_env(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _interpolate_env(v) for k, v in obj.items()}
    return obj


def _nearest_existing_parent(path: Path) -> Path:
    current = path.expanduser()
    if not current.is_absolute():
        current = Path.cwd() / current
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def get_free_disk_gb(path: Path) -> float:
    """Return available disk space (GiB) on the filesystem hosting ``path``."""
    anchor = _nearest_existing_parent(path)
    usage = shutil.disk_usage(anchor)
    return usage.free / (1024**3)


def check_storage_guard(path: Path, min_free_gb: float = 5.0) -> float:
    """Verify the target filesystem has at least ``min_free_gb`` GiB available."""
    free_gb = get_free_disk_gb(path)
    if free_gb < min_free_gb:
        raise StorageGuardError(
            f"Insufficient free disk space at {path} ({free_gb:.2f} GiB < {min_free_gb:.2f} GiB). "
            "Store caches, models, and runs under /mnt/data0 instead of the root filesystem."
        )
    return free_gb


def load_experiment_config(
    config_path: Path,
    dotenv_path: Path | None = None,
    enforce_storage_guard: bool = False,
) -> ExperimentConfig:
    """Load and validate an ``ExperimentConfig`` from a YAML file."""
    if dotenv_path is not None:
        load_dotenv_file(dotenv_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    expanded = _interpolate_env(raw)
    cfg = ExperimentConfig.model_validate(expanded)

    if enforce_storage_guard:
        min_gb = cfg.guardrails.min_free_disk_gb
        check_storage_guard(Path(cfg.experiment.workspace_root), min_free_gb=min_gb)
        check_storage_guard(Path(cfg.experiment.output_dir), min_free_gb=min_gb)

    return cfg
