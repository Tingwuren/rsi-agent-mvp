"""Environment diagnostics for ``rsi doctor``."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rsi_mvp.config import get_free_disk_gb

CheckStatus = Literal["pass", "warn", "fail"]


class DoctorCheck(BaseModel):
    name: str
    status: CheckStatus
    summary: str
    details: str = ""


class GPUInfo(BaseModel):
    index: int
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_pct: int

    @property
    def memory_free_mib(self) -> int:
        return max(0, self.memory_total_mib - self.memory_used_mib)

    @property
    def is_idle(self) -> bool:
        return self.memory_used_mib < 1024 and self.utilization_pct < 10


class DoctorReport(BaseModel):
    checks: list[DoctorCheck] = Field(default_factory=list)
    gpus: list[GPUInfo] = Field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return any(c.status == "fail" for c in self.checks)


def check_python_version() -> DoctorCheck:
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 11):
        return DoctorCheck(name="python_version", status="pass", summary=f"Python {ver_str}")
    return DoctorCheck(
        name="python_version",
        status="fail",
        summary=f"Python {ver_str} (< 3.11 required)",
    )


def check_disk_storage(workspace_root: Path, min_free_gb: float = 20.0) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    ws_free = get_free_disk_gb(workspace_root)
    if ws_free >= min_free_gb:
        checks.append(
            DoctorCheck(
                name="workspace_disk",
                status="pass",
                summary=f"{workspace_root} has {ws_free:.1f} GiB free",
            )
        )
    elif ws_free >= 5.0:
        checks.append(
            DoctorCheck(
                name="workspace_disk",
                status="warn",
                summary=f"{workspace_root} has {ws_free:.1f} GiB free (< {min_free_gb:.0f} GiB)",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="workspace_disk",
                status="fail",
                summary=f"{workspace_root} critically low: {ws_free:.2f} GiB free",
                details="Point RSI_WORKSPACE_ROOT and output_dir to /mnt/data0.",
            )
        )

    etc_free = get_free_disk_gb(Path("/etc"))
    if etc_free < 5.0:
        checks.append(
            DoctorCheck(
                name="root_partition_guard",
                status="warn",
                summary=f"System disk (/etc) has only {etc_free:.2f} GiB free",
                details="Keep HF_HOME, PIP_CACHE_DIR, TMPDIR, and venv under /mnt/data0.",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="root_partition_guard",
                status="pass",
                summary=f"System disk (/etc) has {etc_free:.1f} GiB free",
            )
        )
    return checks


def check_cache_env_vars() -> DoctorCheck:
    hf_home = os.environ.get("HF_HOME", "")
    pip_cache = os.environ.get("PIP_CACHE_DIR", "")
    tmpdir = os.environ.get("TMPDIR", "")

    issues: list[str] = []
    for key, val in [("HF_HOME", hf_home), ("PIP_CACHE_DIR", pip_cache), ("TMPDIR", tmpdir)]:
        if not val:
            issues.append(f"{key} unset")
        elif val.startswith("/root") or val.startswith("/tmp"):
            issues.append(f"{key}={val} (should be on /mnt/data0)")

    if not issues:
        return DoctorCheck(
            name="cache_env_vars",
            status="pass",
            summary="HF_HOME, PIP_CACHE_DIR, TMPDIR directed off root disk",
        )
    return DoctorCheck(
        name="cache_env_vars",
        status="warn",
        summary="; ".join(issues),
        details="Source .env or run scripts/setup_env.sh before downloading models.",
    )


def query_nvidia_gpus() -> tuple[DoctorCheck, list[GPUInfo]]:
    if shutil.which("nvidia-smi") is None:
        return (
            DoctorCheck(name="nvidia_gpus", status="warn", summary="nvidia-smi not found"),
            [],
        )

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:
        return (
            DoctorCheck(name="nvidia_gpus", status="warn", summary=f"nvidia-smi error: {exc}"),
            [],
        )

    if proc.returncode != 0:
        return (
            DoctorCheck(
                name="nvidia_gpus",
                status="warn",
                summary=f"nvidia-smi exited {proc.returncode}",
                details=proc.stderr.strip(),
            ),
            [],
        )

    gpus: list[GPUInfo] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_used_mib=int(parts[2]),
                memory_total_mib=int(parts[3]),
                utilization_pct=int(parts[4]),
            )
        )

    idle_ids = [g.index for g in gpus if g.is_idle]
    if not gpus:
        return (
            DoctorCheck(name="nvidia_gpus", status="warn", summary="No GPUs reported"),
            [],
        )
    if idle_ids:
        return (
            DoctorCheck(
                name="nvidia_gpus",
                status="pass",
                summary=f"Detected {len(gpus)} GPUs ({gpus[0].name}); idle GPUs: {idle_ids}",
            ),
            gpus,
        )
    return (
        DoctorCheck(
            name="nvidia_gpus",
            status="warn",
            summary=f"Detected {len(gpus)} GPUs ({gpus[0].name}), but 0 idle GPUs right now",
        ),
        gpus,
    )


def check_torch_cuda(run_matmul: bool = True) -> DoctorCheck:
    if importlib.util.find_spec("torch") is None:
        return DoctorCheck(
            name="torch_cuda",
            status="warn",
            summary="PyTorch not installed in current environment",
            details="Run scripts/setup_env.sh --with-gpu to install Blackwell-compatible PyTorch.",
        )

    import torch

    if not torch.cuda.is_available():
        return DoctorCheck(
            name="torch_cuda",
            status="warn",
            summary=f"PyTorch {torch.__version__} installed, but CUDA unavailable",
        )

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    summary = (
        f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, "
        f"{device_name} (sm_{cap[0]}{cap[1]})"
    )

    if not run_matmul:
        return DoctorCheck(name="torch_cuda", status="pass", summary=summary)

    try:
        a = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        _ = float(c[0, 0].item())
        return DoctorCheck(
            name="torch_cuda",
            status="pass",
            summary=f"{summary} — BF16 matmul OK",
        )
    except Exception as exc:
        return DoctorCheck(
            name="torch_cuda",
            status="fail",
            summary=f"{summary} — CUDA matmul FAILED",
            details=str(exc),
        )


def check_optional_packages() -> DoctorCheck:
    pkgs = [
        "pydantic",
        "yaml",
        "openai",
        "pytest",
        "ruff",
        "vllm",
        "transformers",
        "peft",
        "trl",
        "datasets",
    ]
    present = [p for p in pkgs if importlib.util.find_spec(p) is not None]
    missing = [p for p in pkgs if p not in present]
    status: CheckStatus = "pass" if not missing else "warn"
    return DoctorCheck(
        name="python_packages",
        status=status,
        summary=f"Installed ({len(present)}/{len(pkgs)}): {', '.join(present)}",
        details=f"Missing optional: {', '.join(missing)}" if missing else "",
    )


def check_docker() -> DoctorCheck:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        return DoctorCheck(
            name="docker",
            status="warn",
            summary="docker CLI not found (subprocess sandbox still available)",
        )
    return DoctorCheck(name="docker", status="pass", summary=f"docker CLI at {docker_bin}")


def run_doctor(
    workspace_root: Path,
    run_gpu_matmul: bool = True,
    min_free_gb: float = 20.0,
) -> DoctorReport:
    """Execute all environment checks and return a structured report."""
    checks: list[DoctorCheck] = [check_python_version()]
    checks.extend(check_disk_storage(workspace_root, min_free_gb=min_free_gb))
    checks.append(check_cache_env_vars())
    gpu_check, gpus = query_nvidia_gpus()
    checks.append(gpu_check)
    checks.append(check_torch_cuda(run_matmul=run_gpu_matmul))
    checks.append(check_optional_packages())
    checks.append(check_docker())
    return DoctorReport(checks=checks, gpus=gpus)
