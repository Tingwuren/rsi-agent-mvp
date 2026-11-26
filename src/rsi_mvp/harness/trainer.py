"""Track-B LoRA fine-tuning backends and anti-collapse SFT trajectory exporter."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

from rsi_mvp.harness.adapters import TaskAdapter
from rsi_mvp.schemas import (
    LoRAAdapterRecord,
    TaskSolution,
    TaskSpec,
    VerifyResult,
    WeightAdaptationConfig,
)

_TRAIN_WORKER_PATH = Path(__file__).resolve().parent / "lora_train_worker.py"


def export_sft_dataset(
    evaluated_items: list[tuple[TaskSpec, TaskSolution, VerifyResult]],
    adapter: TaskAdapter,
    output_jsonl: Path,
    min_confidence: float = 0.7,
    replay_jsonl: Path | None = None,
    replay_ratio: float = 0.2,
    seed: int = 42,
) -> int:
    """Export verified trajectories to ShareGPT/messages JSONL with optional Replay mixing.

    Only trajectories with ``verify.passed == True`` and ``verify.confidence >= min_confidence``
    are admitted, preventing noisy or low-consensus label-free samples from poisoning weights.
    """
    fresh_rows: list[dict[str, Any]] = []
    for spec, solution, verify in evaluated_items:
        if not verify.passed:
            continue
        if verify.confidence < min_confidence - 1e-9:
            continue
        formatted = adapter.format_for_sft(spec, solution, verify)
        if formatted.get("messages"):
            fresh_rows.append(formatted)

    if not fresh_rows:
        return 0

    combined = list(fresh_rows)
    if replay_jsonl is not None and replay_jsonl.is_file() and replay_ratio > 0.0:
        replay_rows: list[dict[str, Any]] = []
        with replay_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    replay_rows.append(json.loads(line))
        if replay_rows:
            rng = random.Random(seed)
            rng.shuffle(replay_rows)
            n_replay = max(1, int(len(fresh_rows) * replay_ratio / max(1e-6, 1.0 - replay_ratio)))
            combined.extend(replay_rows[:n_replay])

    rng_out = random.Random(seed)
    rng_out.shuffle(combined)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in combined:
            fh.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
    return len(combined)


class TrainerBackend(Protocol):
    """Protocol for LoRA weight adaptation backends (TRL SFT, Mock, future DPO/GRPO)."""

    def train_lora(
        self,
        adapter_id: str,
        base_model: str,
        parent_adapter_path: str | None,
        train_jsonl: Path,
        output_dir: Path,
        gpu_ids: list[int],
        config: WeightAdaptationConfig,
        seed: int = 42,
    ) -> LoRAAdapterRecord: ...


class TRLLoRASFTTrainer:
    """Runs single/multi-GPU BF16 LoRA SFT inside an isolated child Python process."""

    def __init__(self, workspace_root: Path, max_length: int = 4096) -> None:
        self.workspace_root = Path(workspace_root)
        self.max_length = max_length

    def train_lora(
        self,
        adapter_id: str,
        base_model: str,
        parent_adapter_path: str | None,
        train_jsonl: Path,
        output_dir: Path,
        gpu_ids: list[int],
        config: WeightAdaptationConfig,
        seed: int = 42,
    ) -> LoRAAdapterRecord:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(_TRAIN_WORKER_PATH),
            "--base-model",
            base_model,
            "--train-jsonl",
            str(train_jsonl.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            "--lora-r",
            str(config.lora_r),
            "--lora-alpha",
            str(config.lora_alpha),
            "--learning-rate",
            str(config.learning_rate),
            "--num-epochs",
            str(config.num_epochs),
            "--max-length",
            str(self.max_length),
            "--seed",
            str(seed),
        ]
        if parent_adapter_path:
            cmd.extend(["--parent-adapter", str(parent_adapter_path)])

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        ws = self.workspace_root
        env.setdefault("HF_HOME", str(ws / "hf_cache"))
        env.setdefault("TMPDIR", str(ws / "tmp"))
        env.setdefault("XDG_CACHE_HOME", str(ws / "tmp" / "xdg_cache"))
        env.setdefault("TRITON_CACHE_DIR", str(ws / "tmp" / "triton_cache"))
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ws / "tmp" / "inductor_cache"))

        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"LoRA training failed (exit={proc.returncode}):\n{proc.stderr[-2000:]}"
            )

        metrics_path = output_dir / "train_metrics.json"
        metrics: dict[str, Any] = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.is_file()
            else {"train_samples": 0, "train_loss": None}
        )
        raw_loss = metrics.get("train_loss")
        return LoRAAdapterRecord(
            adapter_id=adapter_id,
            base_model=base_model,
            parent_adapter_id=Path(parent_adapter_path).name if parent_adapter_path else None,
            adapter_path=str(output_dir.resolve()),
            train_samples=int(metrics.get("train_samples") or 0),
            method=config.method,
            train_loss=float(raw_loss) if raw_loss is not None else None,
            accepted=False,
        )


class MockTrainerBackend:
    """Zero-GPU deterministic mock trainer for offline unit and integration tests."""

    def __init__(self, synthetic_loss: float = 0.35) -> None:
        self.synthetic_loss = synthetic_loss
        self.calls: list[dict[str, Any]] = []

    def train_lora(
        self,
        adapter_id: str,
        base_model: str,
        parent_adapter_path: str | None,
        train_jsonl: Path,
        output_dir: Path,
        gpu_ids: list[int],
        config: WeightAdaptationConfig,
        seed: int = 42,
    ) -> LoRAAdapterRecord:
        lines = [
            ln.strip()
            for ln in train_jsonl.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        for ln in lines:
            row = json.loads(ln)
            if "messages" not in row or not isinstance(row["messages"], list):
                raise ValueError("Invalid SFT JSONL row: missing 'messages' list")

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "r": config.lora_r,
                    "lora_alpha": config.lora_alpha,
                    "base_model_name_or_path": base_model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.calls.append(
            {
                "adapter_id": adapter_id,
                "train_samples": len(lines),
                "gpu_ids": list(gpu_ids),
                "seed": seed,
            }
        )
        return LoRAAdapterRecord(
            adapter_id=adapter_id,
            base_model=base_model,
            parent_adapter_id=Path(parent_adapter_path).name if parent_adapter_path else None,
            adapter_path=str(output_dir.resolve()),
            train_samples=len(lines),
            method=config.method,
            train_loss=self.synthetic_loss,
            accepted=False,
        )
