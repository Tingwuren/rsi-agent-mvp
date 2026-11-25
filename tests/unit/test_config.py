"""Unit tests for configuration loading, env interpolation, and storage guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsi_mvp.config import (
    StorageGuardError,
    check_storage_guard,
    load_dotenv_file,
    load_experiment_config,
)


def test_load_experiment_config_interpolates_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_WORKSPACE_ROOT", str(tmp_path / "ws"))
    cfg_path = Path("configs/code_mbpp_plus.yaml")
    cfg = load_experiment_config(cfg_path)
    assert cfg.experiment.workspace_root == str(tmp_path / "ws")
    assert cfg.experiment.output_dir == str(tmp_path / "ws/runs")
    assert cfg.model.base_model == "Qwen/Qwen3-8B"
    assert cfg.hardware.serving_gpus == [1]


def test_load_dotenv_file_respects_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nRSI_TEST_KEY=from_file\nRSI_EXISTING=new_val\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("RSI_TEST_KEY", raising=False)
    monkeypatch.setenv("RSI_EXISTING", "old_val")

    loaded = load_dotenv_file(env_file, override=False)
    assert loaded == {"RSI_TEST_KEY": "from_file"}
    assert "RSI_EXISTING" not in loaded

    loaded_override = load_dotenv_file(env_file, override=True)
    assert loaded_override["RSI_EXISTING"] == "new_val"


def test_storage_guard_blocks_low_space_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rsi_mvp.config.get_free_disk_gb", lambda _p: 0.6)
    with pytest.raises(StorageGuardError, match="Insufficient free disk space"):
        check_storage_guard(tmp_path, min_free_gb=5.0)
