"""Unit tests for Track-B SFT exporter, Replay anti-collapse, Trainer, and LifecycleManager."""

from __future__ import annotations

import json
from pathlib import Path

from rsi_mvp.cli import main as cli_main
from rsi_mvp.harness.adapters import get_adapter
from rsi_mvp.harness.lifecycle import MockLifecycleManager, VLLMLifecycleManager
from rsi_mvp.harness.trainer import MockTrainerBackend, export_sft_dataset
from rsi_mvp.schemas import (
    CandidateOutput,
    ExperimentConfig,
    TaskSolution,
    TaskSpec,
    VerifyResult,
    WeightAdaptationConfig,
)


def test_export_sft_dataset_filters_confidence_and_mixes_replay(tmp_path: Path) -> None:
    adapter = get_adapter("code")
    specs = [
        TaskSpec(task_id="t_good", domain="code", split="train", prompt="Write f1"),
        TaskSpec(task_id="t_low_conf", domain="code", split="train", prompt="Write f2"),
        TaskSpec(task_id="t_failed", domain="code", split="train", prompt="Write f3"),
    ]
    sols = [
        TaskSolution(
            task_id=s.task_id,
            outputs=[
                CandidateOutput(raw_text="def f(): return 1", extracted_answer="def f(): return 1")
            ],
        )
        for s in specs
    ]
    vers = [
        VerifyResult(task_id="t_good", passed=True, score=1.0, confidence=0.9),
        VerifyResult(task_id="t_low_conf", passed=True, score=1.0, confidence=0.4),
        VerifyResult(task_id="t_failed", passed=False, score=0.0, confidence=1.0),
    ]

    replay_file = tmp_path / "replay.jsonl"
    replay_row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "replay_q"},
            {"role": "assistant", "content": "replay_a"},
        ]
    }
    replay_file.write_text(json.dumps(replay_row) + "\n", encoding="utf-8")

    out_jsonl = tmp_path / "train_sft.jsonl"
    count = export_sft_dataset(
        evaluated_items=list(zip(specs, sols, vers, strict=True)),
        adapter=adapter,
        output_jsonl=out_jsonl,
        min_confidence=0.7,
        replay_jsonl=replay_file,
        replay_ratio=0.5,
        seed=42,
    )
    # Only 1 fresh item passes (t_good) + 1 replay item = 2 total
    assert count == 2
    rows = [json.loads(ln) for ln in out_jsonl.read_text(encoding="utf-8").splitlines()]
    user_msgs = {r["messages"][1]["content"] for r in rows}
    assert user_msgs == {"Write f1", "replay_q"}


def test_vllm_lifecycle_command_builder_includes_lora_and_data0_caches(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.experiment.workspace_root = str(tmp_path)
    mgr = VLLMLifecycleManager(config=cfg, model_path_override="/mnt/data0/Qwen3-8B")
    cmd, env = mgr.build_serve_command(lora_modules={"lora_0001": "/mnt/data0/adapters/lora_0001"})
    assert "--enable-lora" in cmd
    assert "--lora-modules" in cmd
    assert "lora_0001=/mnt/data0/adapters/lora_0001" in cmd
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"
    assert str(tmp_path) in env["TRITON_CACHE_DIR"]


def test_mock_trainer_and_cli_train_lora(tmp_path: Path) -> None:
    sft_file = tmp_path / "sft.jsonl"
    sft_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trainer = MockTrainerBackend(synthetic_loss=0.28)
    rec = trainer.train_lora(
        adapter_id="lora_0001",
        base_model="Qwen/Qwen3-8B",
        parent_adapter_path=None,
        train_jsonl=sft_file,
        output_dir=tmp_path / "adapters" / "lora_0001",
        gpu_ids=[1],
        config=WeightAdaptationConfig(lora_r=16, lora_alpha=32),
    )
    assert rec.adapter_id == "lora_0001"
    assert rec.train_samples == 1
    assert rec.train_loss == 0.28
    assert (Path(rec.adapter_path) / "adapter_config.json").is_file()

    # Verify CLI `rsi train-lora --mock`
    code = cli_main(
        [
            "train-lora",
            "--config",
            "configs/code_mbpp_plus.yaml",
            "--train-jsonl",
            str(sft_file),
            "--adapter-id",
            "lora_cli_test",
            "--output",
            str(tmp_path / "cli_adapter"),
            "--mock",
        ]
    )
    assert code == 0
    assert (tmp_path / "cli_adapter" / "adapter_config.json").is_file()

    # Verify MockLifecycleManager transition sequence
    lc = MockLifecycleManager(training_gpus=[1])
    lc.ensure_serving(None)
    assert lc.acquire_for_training() == [1]
    lc.release_after_training(rec)
    lc.shutdown()
    assert lc.events == [
        "ensure_serving:base",
        "acquire_for_training",
        "release_after_training:lora_0001",
        "shutdown",
    ]
