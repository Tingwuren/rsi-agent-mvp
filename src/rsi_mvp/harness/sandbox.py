"""Isolated sandbox executors: SubprocessSandbox and DockerSandbox."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from rsi_mvp.schemas import SandboxConfig

_WORKER_ENTRY_PATH = Path(__file__).resolve().parent / "worker_entry.py"
_MAX_STDIO_CHARS = 8192


def _truncate(text: str, limit: int = _MAX_STDIO_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2 - 20
    return f"{text[:half]}\n...[truncated]...\n{text[-half:]}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LLMBackendSpec(BaseModel):
    """Serializable LLM connection configuration passed into sandbox workers."""

    mode: Literal["mock", "openai_http"] = "mock"
    mock_responses: list[str] = Field(default_factory=list)
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "mock-llm"
    timeout_sec: float = 120.0


class SandboxResult(BaseModel):
    """Structured outcome returned by any ``SandboxExecutor`` method."""

    ok: bool
    exit_code: int
    payload: dict[str, Any] | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    tampered: bool = False
    elapsed_sec: float = 0.0


class SandboxExecutor(Protocol):
    """Protocol implemented by SubprocessSandbox and DockerSandbox."""

    def run_agent_solve(
        self,
        snapshot_dir: Path,
        task_input: dict[str, Any],
        llm_spec: LLMBackendSpec,
        timeout_sec: int = 30,
    ) -> SandboxResult: ...

    def run_agent_improve(
        self,
        snapshot_dir: Path,
        improve_context: dict[str, Any],
        llm_spec: LLMBackendSpec,
        timeout_sec: int = 90,
    ) -> SandboxResult: ...

    def run_pytest_eval(
        self,
        solution_code: str,
        entry_file: str,
        test_code: str,
        timeout_sec: int = 20,
    ) -> SandboxResult: ...


def _run_process_group(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_sec: int,
) -> tuple[int, str, str, bool, float]:
    """Execute ``cmd`` in its own POSIX session and kill the entire group on timeout."""
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        exit_code = -9

    elapsed = round(time.perf_counter() - start, 4)
    return exit_code, _truncate(stdout or ""), _truncate(stderr or ""), timed_out, elapsed


class SubprocessSandbox:
    """Lightweight process-group isolated sandbox suitable for local dev and CI."""

    def __init__(self, tmp_root: Path | None = None) -> None:
        self._tmp_root = tmp_root

    def _base_tmp(self) -> str | None:
        if self._tmp_root is not None:
            self._tmp_root.mkdir(parents=True, exist_ok=True)
            return str(self._tmp_root)
        return os.environ.get("TMPDIR")

    def _run_worker(
        self,
        mode: Literal["solve", "improve"],
        snapshot_dir: Path,
        payload: dict[str, Any],
        llm_spec: LLMBackendSpec,
        timeout_sec: int,
    ) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="rsi_sbx_", dir=self._base_tmp()) as tmpdir:
            ipc_dir = Path(tmpdir)
            in_file = ipc_dir / "input.json"
            out_file = ipc_dir / "output.json"
            in_file.write_text(
                json.dumps(
                    {"payload": payload, "llm_spec": llm_spec.model_dump()},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            # Minimal clean env: deliberately omits PYTHONPATH so harness is unreachable
            clean_env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": tmpdir,
                "HOME": tmpdir,
            }
            cmd = [
                sys.executable,
                "-I",  # isolated mode: ignore user site-packages and PYTHONPATH
                str(_WORKER_ENTRY_PATH),
                "--mode",
                mode,
                "--snapshot",
                str(snapshot_dir.resolve()),
                "--input",
                str(in_file),
                "--output",
                str(out_file),
            ]
            exit_code, stdout, stderr, timed_out, elapsed = _run_process_group(
                cmd=cmd,
                cwd=ipc_dir,
                env=clean_env,
                timeout_sec=timeout_sec,
            )

            if timed_out:
                return SandboxResult(
                    ok=False,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=f"Timed out after {timeout_sec}s\n{stderr}".strip(),
                    timed_out=True,
                    elapsed_sec=elapsed,
                )

            if not out_file.is_file():
                return SandboxResult(
                    ok=False,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr or "Worker exited without writing output.json",
                    elapsed_sec=elapsed,
                )

            raw_out = json.loads(out_file.read_text(encoding="utf-8"))
            if raw_out.get("ok") and exit_code == 0:
                return SandboxResult(
                    ok=True,
                    exit_code=0,
                    payload=raw_out.get("result"),
                    stdout=stdout,
                    stderr=stderr,
                    elapsed_sec=elapsed,
                )

            err_msg = raw_out.get("error") or stderr or "Unknown worker error"
            tb = raw_out.get("traceback") or ""
            return SandboxResult(
                ok=False,
                exit_code=exit_code,
                payload=None,
                stdout=stdout,
                stderr=_truncate(f"{err_msg}\n{tb}".strip()),
                elapsed_sec=elapsed,
            )

    def run_agent_solve(
        self,
        snapshot_dir: Path,
        task_input: dict[str, Any],
        llm_spec: LLMBackendSpec,
        timeout_sec: int = 30,
    ) -> SandboxResult:
        return self._run_worker("solve", snapshot_dir, task_input, llm_spec, timeout_sec)

    def run_agent_improve(
        self,
        snapshot_dir: Path,
        improve_context: dict[str, Any],
        llm_spec: LLMBackendSpec,
        timeout_sec: int = 90,
    ) -> SandboxResult:
        return self._run_worker("improve", snapshot_dir, improve_context, llm_spec, timeout_sec)

    def run_pytest_eval(
        self,
        solution_code: str,
        entry_file: str,
        test_code: str,
        timeout_sec: int = 20,
    ) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="rsi_pytest_", dir=self._base_tmp()) as tmpdir:
            work = Path(tmpdir)
            sol_path = work / entry_file
            sol_path.parent.mkdir(parents=True, exist_ok=True)
            sol_path.write_text(solution_code, encoding="utf-8")

            test_path = work / "test_solution.py"
            test_path.write_text(test_code, encoding="utf-8")
            expected_hash = _sha256_file(test_path)
            os.chmod(test_path, 0o444)

            clean_env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(work),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": tmpdir,
                "HOME": tmpdir,
            }
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "--no-header",
                "-o",
                "addopts=",
                str(test_path),
            ]
            try:
                exit_code, stdout, stderr, timed_out, elapsed = _run_process_group(
                    cmd=cmd,
                    cwd=work,
                    env=clean_env,
                    timeout_sec=timeout_sec,
                )
                tampered = (not test_path.is_file()) or (_sha256_file(test_path) != expected_hash)
            finally:
                if test_path.exists():
                    os.chmod(test_path, 0o644)

            if tampered:
                return SandboxResult(
                    ok=False,
                    exit_code=-2,
                    stdout=stdout,
                    stderr="Security violation: test_solution.py was modified during execution.",
                    timed_out=timed_out,
                    tampered=True,
                    elapsed_sec=elapsed,
                )

            return SandboxResult(
                ok=(not timed_out and exit_code == 0),
                exit_code=exit_code,
                stdout=stdout,
                stderr=f"Timed out after {timeout_sec}s\n{stderr}".strip() if timed_out else stderr,
                timed_out=timed_out,
                tampered=False,
                elapsed_sec=elapsed,
            )


class DockerSandbox:
    """Hardened Docker container sandbox (--network none, --read-only, memory/CPU/PID caps)."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "1g",
        cpus: str = "2",
        pids_limit: int = 128,
        tmp_root: Path | None = None,
    ) -> None:
        self.image = image
        self.memory_limit = memory_limit
        self.cpus = cpus
        self.pids_limit = pids_limit
        self._tmp_root = tmp_root

    @staticmethod
    def is_docker_available() -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            proc = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def _base_tmp(self) -> str | None:
        if self._tmp_root is not None:
            self._tmp_root.mkdir(parents=True, exist_ok=True)
            return str(self._tmp_root)
        return os.environ.get("TMPDIR")

    def build_docker_command(
        self,
        container_name: str,
        mounts: list[tuple[Path, str, Literal["ro", "rw"]]],
        inner_cmd: list[str],
        network_mode: str = "none",
        workdir: str = "/work",
    ) -> list[str]:
        cmd: list[str] = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            network_mode,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--memory",
            self.memory_limit,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "HOME=/tmp",
            "-e",
            "TMPDIR=/tmp",
            "-w",
            workdir,
        ]
        for host_path, container_path, mode in mounts:
            cmd.extend(["-v", f"{host_path.resolve()}:{container_path}:{mode}"])
        cmd.append(self.image)
        cmd.extend(inner_cmd)
        return cmd

    def _exec_container(
        self,
        container_name: str,
        docker_cmd: list[str],
        timeout_sec: int,
    ) -> tuple[int, str, str, bool, float]:
        start = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            exit_code = proc.returncode
            stdout = _truncate(proc.stdout or "")
            stderr = _truncate(proc.stderr or "")
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            exit_code = -9
            stdout = ""
            stderr = f"Docker container timed out after {timeout_sec}s"
        elapsed = round(time.perf_counter() - start, 4)
        return exit_code, stdout, stderr, timed_out, elapsed

    def _run_worker(
        self,
        mode: Literal["solve", "improve"],
        snapshot_dir: Path,
        payload: dict[str, Any],
        llm_spec: LLMBackendSpec,
        timeout_sec: int,
    ) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="rsi_dkr_", dir=self._base_tmp()) as tmpdir:
            ipc_dir = Path(tmpdir)
            os.chmod(ipc_dir, 0o777)
            in_file = ipc_dir / "input.json"
            out_file = ipc_dir / "output.json"
            in_file.write_text(
                json.dumps(
                    {"payload": payload, "llm_spec": llm_spec.model_dump()},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.chmod(in_file, 0o644)

            container_name = f"rsi-worker-{uuid.uuid4().hex[:12]}"
            network_mode = "none" if llm_spec.mode == "mock" else "host"
            docker_cmd = self.build_docker_command(
                container_name=container_name,
                mounts=[
                    (_WORKER_ENTRY_PATH, "/runner/worker_entry.py", "ro"),
                    (snapshot_dir / "agent", "/snapshot/agent", "ro"),
                    (ipc_dir, "/ipc", "rw"),
                ],
                inner_cmd=[
                    "python3",
                    "-I",
                    "/runner/worker_entry.py",
                    "--mode",
                    mode,
                    "--snapshot",
                    "/snapshot",
                    "--input",
                    "/ipc/input.json",
                    "--output",
                    "/ipc/output.json",
                ],
                network_mode=network_mode,
                workdir="/tmp",
            )
            exit_code, stdout, stderr, timed_out, elapsed = self._exec_container(
                container_name=container_name,
                docker_cmd=docker_cmd,
                timeout_sec=timeout_sec,
            )
            if timed_out:
                return SandboxResult(
                    ok=False,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=True,
                    elapsed_sec=elapsed,
                )
            if not out_file.is_file():
                return SandboxResult(
                    ok=False,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr or "Container exited without writing output.json",
                    elapsed_sec=elapsed,
                )
            raw_out = json.loads(out_file.read_text(encoding="utf-8"))
            if raw_out.get("ok") and exit_code == 0:
                return SandboxResult(
                    ok=True,
                    exit_code=0,
                    payload=raw_out.get("result"),
                    stdout=stdout,
                    stderr=stderr,
                    elapsed_sec=elapsed,
                )
            return SandboxResult(
                ok=False,
                exit_code=exit_code,
                stdout=stdout,
                stderr=_truncate(f"{raw_out.get('error', '')}\n{raw_out.get('traceback', '')}"),
                elapsed_sec=elapsed,
            )

    def run_agent_solve(
        self,
        snapshot_dir: Path,
        task_input: dict[str, Any],
        llm_spec: LLMBackendSpec,
        timeout_sec: int = 30,
    ) -> SandboxResult:
        return self._run_worker("solve", snapshot_dir, task_input, llm_spec, timeout_sec)

    def run_agent_improve(
        self,
        snapshot_dir: Path,
        improve_context: dict[str, Any],
        llm_spec: LLMBackendSpec,
        timeout_sec: int = 90,
    ) -> SandboxResult:
        return self._run_worker("improve", snapshot_dir, improve_context, llm_spec, timeout_sec)

    def run_pytest_eval(
        self,
        solution_code: str,
        entry_file: str,
        test_code: str,
        timeout_sec: int = 20,
    ) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="rsi_dkr_eval_", dir=self._base_tmp()) as tmpdir:
            work = Path(tmpdir)
            (work / entry_file).write_text(solution_code, encoding="utf-8")
            test_path = work / "test_solution.py"
            test_path.write_text(test_code, encoding="utf-8")
            expected_hash = _sha256_file(test_path)

            container_name = f"rsi-eval-{uuid.uuid4().hex[:12]}"
            # Mount work dir read-only so untrusted solution code cannot tamper with tests
            docker_cmd = self.build_docker_command(
                container_name=container_name,
                mounts=[(work, "/work", "ro")],
                inner_cmd=[
                    "python3",
                    "-m",
                    "pytest",
                    "-q",
                    "--tb=short",
                    "--no-header",
                    "-o",
                    "cache_dir=/tmp/pytest_cache",
                    "-o",
                    "addopts=",
                    "/work/test_solution.py",
                ],
                network_mode="none",
                workdir="/work",
            )
            exit_code, stdout, stderr, timed_out, elapsed = self._exec_container(
                container_name=container_name,
                docker_cmd=docker_cmd,
                timeout_sec=timeout_sec,
            )
            tampered = (not test_path.is_file()) or (_sha256_file(test_path) != expected_hash)
            return SandboxResult(
                ok=(not timed_out and not tampered and exit_code == 0),
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                tampered=tampered,
                elapsed_sec=elapsed,
            )


def create_sandbox(cfg: SandboxConfig, tmp_root: Path | None = None) -> SandboxExecutor:
    """Instantiate the configured sandbox backend."""
    if cfg.backend == "subprocess":
        return SubprocessSandbox(tmp_root=tmp_root)
    if cfg.backend == "docker":
        return DockerSandbox(
            image=cfg.docker_image,
            memory_limit=cfg.memory_limit,
            tmp_root=tmp_root,
        )
    raise ValueError(f"Unsupported sandbox backend: {cfg.backend!r}")
