"""Unit tests for ``rsi doctor`` diagnostics and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

from rsi_mvp.cli import main
from rsi_mvp.doctor import GPUInfo, run_doctor


def test_gpu_info_idle_detection() -> None:
    idle = GPUInfo(
        index=1,
        name="NVIDIA GeForce RTX 5090",
        memory_used_mib=4,
        memory_total_mib=32607,
        utilization_pct=0,
    )
    busy = GPUInfo(
        index=0,
        name="NVIDIA GeForce RTX 5090",
        memory_used_mib=28489,
        memory_total_mib=32607,
        utilization_pct=98,
    )
    assert idle.is_idle is True
    assert idle.memory_free_mib == 32603
    assert busy.is_idle is False


def test_run_doctor_produces_structured_report(tmp_path: Path) -> None:
    report = run_doctor(workspace_root=tmp_path, run_gpu_matmul=False)
    names = {c.name for c in report.checks}
    expected = {"python_version", "workspace_disk", "cache_env_vars", "nvidia_gpus", "torch_cuda"}
    assert expected <= names


def test_cli_doctor_json_output(capsys: object, tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("RSI_WORKSPACE_ROOT", str(tmp_path))  # type: ignore[attr-defined]
    code = main(["doctor", "--json", "--no-gpu-matmul"])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    parsed = json.loads(captured.out)
    assert "checks" in parsed
    assert code in (0, 1)
