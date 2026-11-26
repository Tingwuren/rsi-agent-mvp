"""End-to-end M4 integration tests: Dual-track (Scaffold + LoRA) RSI & Anchor rollback."""

from __future__ import annotations

import json
from pathlib import Path

from rsi_mvp.harness.evaluator import SolveFn, make_sandboxed_solve_fn
from rsi_mvp.harness.lifecycle import MockLifecycleManager
from rsi_mvp.harness.llm import BaseLLMClient, MockLLMClient
from rsi_mvp.harness.loop import EvolutionOrchestrator
from rsi_mvp.harness.sandbox import LLMBackendSpec, SubprocessSandbox
from rsi_mvp.harness.trainer import MockTrainerBackend
from rsi_mvp.schemas import (
    DatasetConfig,
    EvolutionConfig,
    ExperimentConfig,
    ExperimentMetaConfig,
    GuardrailsConfig,
    SandboxConfig,
    TaskConfig,
    TaskSpec,
    WeightAdaptationConfig,
)


def _write_four_split_dataset(ds_dir: Path) -> None:
    ds_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[TaskSpec]] = {
        "train": [
            TaskSpec(
                task_id="tr_1",
                domain="code",
                split="train",
                prompt="Write double(x).",
                payload={"entry_file": "solution.py"},
                ground_truth={
                    "test_code": "from solution import double\ndef test_1(): assert double(2)==4\n"
                },
            ),
            TaskSpec(
                task_id="tr_2",
                domain="code",
                split="train",
                prompt="Write add1(x).",
                payload={"entry_file": "solution.py"},
                ground_truth={
                    "test_code": "from solution import add1\ndef test_2(): assert add1(2)==3\n"
                },
            ),
        ],
        "dev": [
            TaskSpec(
                task_id="dev_1",
                domain="code",
                split="dev",
                prompt="Write double(x).",
                payload={"entry_file": "solution.py"},
                ground_truth={
                    "test_code": "from solution import double\ndef test_1(): assert double(3)==6\n"
                },
            ),
            TaskSpec(
                task_id="dev_2",
                domain="code",
                split="dev",
                prompt="Write triple(x).",
                payload={"entry_file": "solution.py"},
                ground_truth={
                    "test_code": "from solution import triple\ndef test_2(): assert triple(3)==9\n"
                },
            ),
        ],
        "anchor": [
            TaskSpec(
                task_id="anc_1",
                domain="code",
                split="anchor",
                prompt="Write add1(x).",
                payload={"entry_file": "solution.py"},
                ground_truth={
                    "test_code": "from solution import add1\ndef test_a(): assert add1(5)==6\n"
                },
            )
        ],
        "holdout": [
            TaskSpec(
                task_id="ho_1",
                domain="code",
                split="holdout",
                prompt="Write quad(x).",
                payload={"entry_file": "solution.py"},
                ground_truth={
                    "test_code": "from solution import quad\ndef test_h(): assert quad(2)==8\n"
                },
            )
        ],
    }
    for name, items in splits.items():
        (ds_dir / f"{name}.jsonl").write_text(
            "\n".join(t.model_dump_json() for t in items) + "\n", encoding="utf-8"
        )


def _dual_track_config(tmp_path: Path, ds_dir: Path) -> ExperimentConfig:
    ws = tmp_path / "ws"
    (ws / "tmp").mkdir(parents=True, exist_ok=True)
    return ExperimentConfig(
        experiment=ExperimentMetaConfig(
            name="m4_dual_track",
            seed=42,
            workspace_root=str(ws),
            output_dir=str(ws / "runs"),
        ),
        task=TaskConfig(adapter="code", verifier="ground_truth_code"),
        dataset=DatasetConfig(
            provider="jsonl",
            path=str(ds_dir),
            mini_eval_size=2,
            full_eval_top_k=2,
            max_concurrency=1,
        ),
        evolution=EvolutionConfig(
            max_generations=2,
            selection_policy="hill_climbing",
            holdout_eval_interval=2,
            early_stop_patience=4,
        ),
        weight_adaptation=WeightAdaptationConfig(
            enabled=True,
            trigger_every_n_generations=1,
            method="sft_lora",
            min_train_samples=2,
            min_confidence=0.7,
        ),
        guardrails=GuardrailsConfig(anchor_enabled=True, max_anchor_drop=0.03),
        sandbox=SandboxConfig(backend="subprocess", task_timeout_sec=15, improve_timeout_sec=20),
    )


def _noop_improve_factory(_gen: int, _parent: object) -> LLMBackendSpec:
    prop = {
        "diagnosis": "minor prompt tweak",
        "hypothesis": "keep scaffold stable while testing Track-B LoRA",
        "rationale": "control scaffold track",
        "edits": [
            {
                "target_path": "prompts.py",
                "mode": "replace_block",
                "search": "Do not omit required imports or function definitions.",
                "replace": "Do not omit required imports, helpers, or function definitions.",
            }
        ],
    }
    return LLMBackendSpec(mode="mock", mock_responses=[json.dumps(prop)])


def test_tc_train_01_dual_track_lora_accept_and_lineage(tmp_path: Path) -> None:
    """TC-TRAIN-01: Single-GPU timeshare stops vLLM, trains LoRA, hot-loads, and improves Dev."""
    ds_dir = tmp_path / "ds"
    _write_four_split_dataset(ds_dir)
    sandbox = SubprocessSandbox(tmp_root=tmp_path / "ws" / "tmp")
    lifecycle = MockLifecycleManager(training_gpus=[1])
    trainer = MockTrainerBackend(synthetic_loss=0.31)
    cfg = _dual_track_config(tmp_path, ds_dir)

    orch_ref: list[EvolutionOrchestrator] = []

    def _solve_factory(_gen: int, snap_dir: Path) -> tuple[SolveFn, BaseLLMClient]:
        has_lora = orch_ref[0].active_lora is not None

        def _dyn_solve(task_input: dict[str, object], _client: BaseLLMClient) -> dict[str, object]:
            tid = str(task_input["task_id"])
            if tid in ("tr_1", "dev_1"):
                code = "```python\ndef double(x):\n    return x * 2\n```"
            elif tid in ("tr_2", "anc_1"):
                code = "```python\ndef add1(x):\n    return x + 1\n```"
            elif tid == "dev_2":
                # Base model fails dev_2; LoRA fixes it!
                code = (
                    "```python\ndef triple(x):\n    return x * 3\n```"
                    if has_lora
                    else "```python\ndef triple(x):\n    return x * 2\n```"
                )
            else:
                code = (
                    "```python\ndef quad(x):\n    return x * 4\n```"
                    if has_lora
                    else "```python\ndef quad(x):\n    return x * 2\n```"
                )
            return make_sandboxed_solve_fn(
                snapshot_dir=snap_dir,
                sandbox=sandbox,
                llm_spec=LLMBackendSpec(mode="mock", mock_responses=[code]),
                timeout_sec=15,
            )(task_input, _client)

        return _dyn_solve, MockLLMClient()

    orch = EvolutionOrchestrator(
        config=cfg,
        run_dir=tmp_path / "ws" / "runs" / "dual_ok",
        sandbox=sandbox,
        solve_factory=_solve_factory,
        improve_spec_factory=_noop_improve_factory,
        lifecycle=lifecycle,
        trainer=trainer,
        enforce_storage_guard=False,
    )
    orch_ref.append(orch)

    cands = orch.run(max_generations=1)
    lora_cands = [c for c in cands if c.track == "weight_lora"]
    assert len(lora_cands) == 1
    lcand = lora_cands[0]
    assert lcand.status == "accepted"
    assert lcand.lora_adapter_id == "lora_0001"
    assert lcand.dev_full_score == 1.0
    assert lcand.anchor_score == 1.0
    assert lcand.holdout_score == 1.0

    # Verify strict single-GPU time-sharing sequence: acquire -> train -> release(lora_0001)
    assert lifecycle.events == [
        "acquire_for_training",
        "release_after_training:lora_0001",
    ]
    assert len(trainer.calls) == 1
    assert trainer.calls[0]["train_samples"] == 2
    adapters = orch.archive.list_lora_adapters()
    assert len(adapters) == 1 and adapters[0].accepted is True


def test_sec_08_lora_anchor_drop_triggers_automatic_rollback(tmp_path: Path) -> None:
    """SEC-08 (Track B): LoRA that degrades Anchor > 3% is rejected and rolled back."""
    ds_dir = tmp_path / "ds_rb"
    _write_four_split_dataset(ds_dir)
    sandbox = SubprocessSandbox(tmp_root=tmp_path / "ws" / "tmp")
    lifecycle = MockLifecycleManager(training_gpus=[1])
    trainer = MockTrainerBackend(synthetic_loss=0.19)
    cfg = _dual_track_config(tmp_path, ds_dir)

    orch_ref: list[EvolutionOrchestrator] = []

    def _solve_factory(_gen: int, snap_dir: Path) -> tuple[SolveFn, BaseLLMClient]:
        has_lora = orch_ref[0].active_lora is not None

        def _dyn_solve(task_input: dict[str, object], _client: BaseLLMClient) -> dict[str, object]:
            tid = str(task_input["task_id"])
            if tid in ("tr_1", "dev_1"):
                code = "```python\ndef double(x):\n    return x * 2\n```"
            elif tid == "tr_2":
                code = "```python\ndef add1(x):\n    return x + 1\n```"
            elif tid == "dev_2":
                # Collapsed LoRA aces dev_2...
                code = (
                    "```python\ndef triple(x):\n    return x * 3\n```"
                    if has_lora
                    else "```python\ndef triple(x):\n    return x * 2\n```"
                )
            elif tid == "anc_1":
                # ...but collapses on Anchor (drops 1.0 -> 0.0)!
                code = (
                    "```python\ndef add1(x):\n    return 999\n```"
                    if has_lora
                    else "```python\ndef add1(x):\n    return x + 1\n```"
                )
            else:
                code = "```python\ndef quad(x):\n    return x * 4\n```"
            return make_sandboxed_solve_fn(
                snapshot_dir=snap_dir,
                sandbox=sandbox,
                llm_spec=LLMBackendSpec(mode="mock", mock_responses=[code]),
                timeout_sec=15,
            )(task_input, _client)

        return _dyn_solve, MockLLMClient()

    orch = EvolutionOrchestrator(
        config=cfg,
        run_dir=tmp_path / "ws" / "runs" / "dual_rollback",
        sandbox=sandbox,
        solve_factory=_solve_factory,
        improve_spec_factory=_noop_improve_factory,
        lifecycle=lifecycle,
        trainer=trainer,
        enforce_storage_guard=False,
    )
    orch_ref.append(orch)

    cands = orch.run(max_generations=1)
    lora_cands = [c for c in cands if c.track == "weight_lora"]
    assert len(lora_cands) == 1
    assert lora_cands[0].status == "rejected_anchor_drop"
    # Active LoRA MUST be rolled back to None (base) and lifecycle restored to base!
    assert orch.active_lora is None
    assert lifecycle.events == [
        "acquire_for_training",
        "release_after_training:lora_0001",
        "ensure_serving:base",
    ]
    adapters = orch.archive.list_lora_adapters()
    assert len(adapters) == 1 and adapters[0].accepted is False
