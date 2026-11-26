"""ModelLifecycleManager: single-GPU time-sharing, multi-GPU split, and vLLM LoRA hot-loading."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from rsi_mvp.harness.sandbox import LLMBackendSpec
from rsi_mvp.schemas import ExperimentConfig, LoRAAdapterRecord


class ModelLifecycleManager(Protocol):
    """Coordinates GPU ownership between vLLM serving and LoRA training."""

    def ensure_serving(
        self, active_lora: LoRAAdapterRecord | None = None
    ) -> LLMBackendSpec: ...

    def acquire_for_training(self) -> list[int]: ...

    def release_after_training(
        self, new_lora: LoRAAdapterRecord | None = None
    ) -> None: ...

    def shutdown(self) -> None: ...


def query_gpu_used_mib(gpu_ids: list[int]) -> dict[int, int]:
    """Query current VRAM usage (MiB) for specific GPU indices via ``nvidia-smi``."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}

    wanted = set(gpu_ids)
    usage: dict[int, int] = {}
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            idx = int(parts[0])
            if idx in wanted:
                usage[idx] = int(parts[1])
    return usage


def wait_for_gpu_memory_release(
    gpu_ids: list[int],
    max_used_mib: int = 2048,
    timeout_sec: int = 60,
    poll_interval_sec: float = 2.0,
) -> bool:
    """Wait until all target GPUs drop below ``max_used_mib`` VRAM usage."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        usage = query_gpu_used_mib(gpu_ids)
        if not usage or all(v <= max_used_mib for v in usage.values()):
            return True
        time.sleep(poll_interval_sec)
    return False


class VLLMLifecycleManager:
    """Manages a local vLLM server with dynamic LoRA support across single/multi-GPU modes."""

    def __init__(
        self,
        config: ExperimentConfig,
        model_path_override: str | None = None,
        gpu_memory_utilization: float = 0.80,
        log_dir: Path | None = None,
        startup_timeout_sec: int = 180,
    ) -> None:
        self.config = config
        self.model_path = model_path_override or config.model.base_model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.log_dir = Path(log_dir or (Path(config.experiment.workspace_root) / "tmp"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout_sec = startup_timeout_sec

        self._proc: subprocess.Popen[bytes] | None = None
        self._log_fh: Any = None
        self._loaded_loras: dict[str, str] = {}

    @property
    def base_url(self) -> str:
        if self.config.hardware.mode == "external_api":
            return os.environ.get(
                self.config.model.base_url_env,
                f"http://{self.config.model.vllm_host}:{self.config.model.vllm_port}/v1",
            )
        return f"http://{self.config.model.vllm_host}:{self.config.model.vllm_port}/v1"

    def _is_healthy(self) -> bool:
        try:
            url = f"{self.base_url.rstrip('/')}/models"
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                return int(resp.status) == 200
        except Exception:
            return False

    def build_serve_command(
        self, lora_modules: dict[str, str] | None = None
    ) -> tuple[list[str], dict[str, str]]:
        cfg = self.config
        cmd: list[str] = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_path,
            "--served-model-name",
            cfg.model.base_model,
            "--host",
            cfg.model.vllm_host,
            "--port",
            str(cfg.model.vllm_port),
            "--max-model-len",
            str(cfg.model.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        if cfg.model.enable_lora:
            cmd.append("--enable-lora")
            if lora_modules:
                cmd.append("--lora-modules")
                for name, path in sorted(lora_modules.items()):
                    cmd.append(f"{name}={path}")
        if len(cfg.hardware.serving_gpus) > 1:
            cmd.extend(["--tensor-parallel-size", str(len(cfg.hardware.serving_gpus))])

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in cfg.hardware.serving_gpus)
        ws = Path(cfg.experiment.workspace_root)
        env.setdefault("HF_HOME", str(ws / "hf_cache"))
        env.setdefault("TMPDIR", str(ws / "tmp"))
        env.setdefault("XDG_CACHE_HOME", str(ws / "tmp" / "xdg_cache"))
        env.setdefault("TRITON_CACHE_DIR", str(ws / "tmp" / "triton_cache"))
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ws / "tmp" / "inductor_cache"))
        env.setdefault("FLASHINFER_WORKSPACE_DIR", str(ws / "tmp" / "flashinfer_cache"))
        env.setdefault("VLLM_NO_USAGE_STATS", "1")
        env.setdefault("VLLM_CACHE_ROOT", str(ws / "tmp" / "vllm_cache"))
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        return cmd, env

    def _try_hot_load_lora(self, lora: LoRAAdapterRecord) -> bool:
        """Attempt zero-downtime dynamic LoRA registration via vLLM runtime API."""
        url = f"{self.base_url.rstrip('/')}/load_lora_adapter"
        payload = json.dumps(
            {"lora_name": lora.adapter_id, "lora_path": lora.adapter_path}
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                if int(resp.status) in (200, 201):
                    self._loaded_loras[lora.adapter_id] = lora.adapter_path
                    return True
        except Exception:
            return False
        return False

    def _start_server(self) -> None:
        cmd, env = self.build_serve_command(lora_modules=self._loaded_loras or None)
        log_file = self.log_dir / "vllm_lifecycle.log"
        self._log_fh = log_file.open("ab")
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.monotonic() + self.startup_timeout_sec
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited early (code={self._proc.returncode}). Check {log_file}"
                )
            if self._is_healthy():
                return
            time.sleep(2.0)

        self.shutdown()
        raise TimeoutError(
            f"vLLM failed to become ready within {self.startup_timeout_sec}s. Check {log_file}"
        )

    def ensure_serving(
        self, active_lora: LoRAAdapterRecord | None = None
    ) -> LLMBackendSpec:
        api_key = os.environ.get(self.config.model.api_key_env, "EMPTY")
        if self.config.hardware.mode == "external_api":
            return LLMBackendSpec(
                mode="openai_http",
                base_url=self.base_url,
                api_key=api_key,
                model=active_lora.adapter_id if active_lora else self.config.model.base_model,
            )

        if active_lora is not None:
            self._loaded_loras[active_lora.adapter_id] = active_lora.adapter_path

        if not self._is_healthy():
            self._start_server()
        elif active_lora is not None:
            # Server already running: attempt runtime hot-load; restart cleanly if unsupported
            if not self._try_hot_load_lora(active_lora) and self._proc is not None:
                self.shutdown()
                wait_for_gpu_memory_release(self.config.hardware.serving_gpus)
                self._start_server()

        return LLMBackendSpec(
            mode="openai_http",
            base_url=self.base_url,
            api_key=api_key,
            model=active_lora.adapter_id if active_lora else self.config.model.base_model,
        )

    def acquire_for_training(self) -> list[int]:
        train_gpus = list(self.config.hardware.training_gpus)
        serving_gpus = set(self.config.hardware.serving_gpus)
        shares_gpu = bool(serving_gpus & set(train_gpus))

        if self.config.hardware.mode == "single_gpu_timeshare" or shares_gpu:
            self.shutdown()
            wait_for_gpu_memory_release(train_gpus, max_used_mib=2048, timeout_sec=60)
        return train_gpus

    def release_after_training(
        self, new_lora: LoRAAdapterRecord | None = None
    ) -> None:
        if new_lora is not None:
            self._loaded_loras[new_lora.adapter_id] = new_lora.adapter_path
        if self.config.hardware.mode != "external_api":
            self.ensure_serving(active_lora=new_lora)

    def shutdown(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=10)
            self._proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None


class MockLifecycleManager:
    """Zero-GPU lifecycle manager recording exact single/multi-GPU transition sequences."""

    def __init__(self, training_gpus: list[int] | None = None) -> None:
        self.training_gpus = list(training_gpus or [1])
        self.events: list[str] = []
        self.active_lora: LoRAAdapterRecord | None = None
        self.serving: bool = False

    def ensure_serving(
        self, active_lora: LoRAAdapterRecord | None = None
    ) -> LLMBackendSpec:
        self.serving = True
        self.active_lora = active_lora
        tag = active_lora.adapter_id if active_lora else "base"
        self.events.append(f"ensure_serving:{tag}")
        return LLMBackendSpec(mode="mock", model=tag)

    def acquire_for_training(self) -> list[int]:
        self.serving = False
        self.events.append("acquire_for_training")
        return list(self.training_gpus)

    def release_after_training(
        self, new_lora: LoRAAdapterRecord | None = None
    ) -> None:
        self.serving = True
        self.active_lora = new_lora
        tag = new_lora.adapter_id if new_lora else "base"
        self.events.append(f"release_after_training:{tag}")

    def shutdown(self) -> None:
        self.serving = False
        self.events.append("shutdown")
