"""End-to-end M3 integration tests: 3-gen RSI loop, resume, funnel, and CLI inspect/diff."""

from __future__ import annotations

import json
from pathlib import Path

from rsi_mvp.cli import main as cli_main
from rsi_mvp.harness.archive import ArchiveStore
from rsi_mvp.harness.evaluator import SolveFn, make_sandboxed_solve_fn
from rsi_mvp.harness.llm import BaseLLMClient, MockLLMClient
from rsi_mvp.harness.loop import EvolutionOrchestrator
from rsi_mvp.harness.sandbox import LLMBackendSpec, SubprocessSandbox
from rsi_mvp.schemas import (
    DatasetConfig,
    EvolutionConfig,
    ExperimentConfig,
    ExperimentMetaConfig,
    GuardrailsConfig,
    SandboxConfig,
    TaskConfig,
    TaskSpec,
)


def _write_mini_code_dataset(ds_dir: Path) -> None:
    ds_dir.mkdir(parents=True, exist_ok=True)
    dev_tasks = [
        TaskSpec(
            task_id="dev_1",
            domain="code",
            split="dev",
            category="math",
            prompt="Write double(x).",
            payload={"entry_file": "solution.py"},
            ground_truth={
                "test_code": "from solution import double\ndef test_1(): assert double(4) == 8\n"
            },
        ),
        TaskSpec(
            task_id="dev_2",
            domain="code",
            split="dev",
            category="math",
            prompt="Write triple(x).",
            payload={"entry_file": "solution.py"},
            ground_truth={
                "test_code": "from solution import triple\ndef test_2(): assert triple(4) == 12\n"
            },
        ),
    ]
    holdout_tasks = [
        TaskSpec(
            task_id="ho_1",
            domain="code",
            split="holdout",
            category="math",
            prompt="Write quad(x).",
            payload={"entry_file": "solution.py"},
            ground_truth={
                "test_code": "from solution import quad\ndef test_h(): assert quad(3) == 12\n"
            },
        )
    ]
    anchor_tasks = [
        TaskSpec(
            task_id="anc_1",
            domain="code",
            split="anchor",
            category="math",
            prompt="Write add1(x).",
            payload={"entry_file": "solution.py"},
            ground_truth={
                "test_code": "from solution import add1\ndef test_a(): assert add1(5) == 6\n"
            },
        )
    ]
    for name, items in [("dev", dev_tasks), ("holdout", holdout_tasks), ("anchor", anchor_tasks)]:
        (ds_dir / f"{name}.jsonl").write_text(
            "\n".join(t.model_dump_json() for t in items) + "\n",
            encoding="utf-8",
        )


def _make_config(tmp_path: Path, ds_dir: Path, max_gens: int = 3) -> ExperimentConfig:
    ws = tmp_path / "ws"
    (ws / "tmp").mkdir(parents=True, exist_ok=True)
    return ExperimentConfig(
        experiment=ExperimentMetaConfig(
            name="m3_e2e",
            seed=42,
            workspace_root=str(ws),
            output_dir=str(ws / "runs"),
        ),
        task=TaskConfig(adapter="code", verifier="ground_truth_code", num_samples_per_task=1),
        dataset=DatasetConfig(
            provider="jsonl",
            path=str(ds_dir),
            mini_eval_size=2,
            full_eval_top_k=2,
            max_concurrency=1,
        ),
        evolution=EvolutionConfig(
            max_generations=max_gens,
            selection_policy="hill_climbing",
            holdout_eval_interval=2,
            early_stop_patience=5,
        ),
        guardrails=GuardrailsConfig(
            anchor_enabled=True,
            max_anchor_drop=0.05,
            max_total_tokens=1_000_000,
        ),
        sandbox=SandboxConfig(backend="subprocess", task_timeout_sec=15, improve_timeout_sec=20),
    )


def test_tc_e2e_01_and_02_evolution_loop_resume_and_cli(tmp_path: Path, capsys: object) -> None:
    """TC-E2E-01 & TC-E2E-02: Gen1 syntax fail, Gen2 improves 0.5->1.0, resume Gen3, CLI works."""
    ds_dir = tmp_path / "ds"
    _write_mini_code_dataset(ds_dir)
    sandbox = SubprocessSandbox(tmp_root=tmp_path / "ws" / "tmp")

    # Solve behavior depends on whether the snapshot contains our evolved marker in prompts.py
    def _solve_factory(_gen: int, snap_dir: Path) -> tuple[SolveFn, BaseLLMClient]:
        prompts_src = (snap_dir / "agent" / "prompts.py").read_text(encoding="utf-8")
        is_evolved = "EVOLVED_V2" in prompts_src

        def _dyn_solve(task_input: dict[str, object], _client: BaseLLMClient) -> dict[str, object]:
            tid = str(task_input["task_id"])
            if tid == "dev_1":
                code = "```python\ndef double(x):\n    return x * 2\n```"
            elif tid == "dev_2":
                # v0 gets dev_2 wrong; evolved snapshot fixes it!
                code = (
                    "```python\ndef triple(x):\n    return x * 3\n```"
                    if is_evolved
                    else "```python\ndef triple(x):\n    return x * 2\n```"
                )
            elif tid == "ho_1":
                code = (
                    "```python\ndef quad(x):\n    return x * 4\n```"
                    if is_evolved
                    else "```python\ndef quad(x):\n    return x * 2\n```"
                )
            else:
                code = "```python\ndef add1(x):\n    return x + 1\n```"

            sbx_solve = make_sandboxed_solve_fn(
                snapshot_dir=snap_dir,
                sandbox=sandbox,
                llm_spec=LLMBackendSpec(mode="mock", mock_responses=[code]),
                timeout_sec=15,
            )
            return sbx_solve(task_input, _client)

        return _dyn_solve, MockLLMClient()

    # Gen 1: introduces broken Python syntax -> rejected_invalid
    # Gen 2: cleanly edits prompts.py adding EVOLVED_V2 -> accepted, dev 0.5 -> 1.0, holdout 0 -> 1
    # Gen 3 (after resume): edits prompts.py benignly -> tie score -> rejected_low_score
    def _improve_spec_factory(gen: int, _parent: object) -> LLMBackendSpec:
        if gen == 1:
            prop = {
                "diagnosis": "Try rewrite",
                "hypothesis": "Introduce syntax error test",
                "rationale": "Testing invalid syntax rejection",
                "edits": [
                    {
                        "target_path": "tools.py",
                        "mode": "overwrite_file",
                        "replace": "def broken(:\n    pass\n",
                    }
                ],
            }
        elif gen == 2:
            prop = {
                "diagnosis": "Solver under-multiplied on triple/quad.",
                "hypothesis": "Add EVOLVED_V2 multiplication verification step.",
                "rationale": "Generalizes across multiplication tasks.",
                "edits": [
                    {
                        "target_path": "prompts.py",
                        "mode": "replace_block",
                        "search": "Carefully read the specification and edge cases.",
                        "replace": "EVOLVED_V2: Carefully read the specification and edge cases.",
                    }
                ],
            }
        else:
            prop = {
                "diagnosis": "Minor wording tweak",
                "hypothesis": "No-op wording polish",
                "rationale": "Test resume generation 3",
                "edits": [
                    {
                        "target_path": "prompts.py",
                        "mode": "replace_block",
                        "search": "EVOLVED_V2:",
                        "replace": "EVOLVED_V2 (polished):",
                    }
                ],
            }
        return LLMBackendSpec(mode="mock", mock_responses=[json.dumps(prop)])

    run_dir = tmp_path / "ws" / "runs" / "exp_resume_test"
    cfg = _make_config(tmp_path, ds_dir, max_gens=2)
    orch1 = EvolutionOrchestrator(
        config=cfg,
        run_dir=run_dir,
        sandbox=sandbox,
        solve_factory=_solve_factory,
        improve_spec_factory=_improve_spec_factory,
        enforce_storage_guard=False,
    )
    cands_after_2 = orch1.run(max_generations=2)
    assert [c.generation for c in cands_after_2] == [0, 1, 2]
    assert cands_after_2[0].status == "accepted"
    assert cands_after_2[0].dev_full_score == 0.5
    assert cands_after_2[0].holdout_score == 0.0
    assert cands_after_2[1].status == "rejected_invalid"
    assert cands_after_2[2].status == "accepted"
    assert cands_after_2[2].dev_full_score == 1.0
    assert cands_after_2[2].holdout_score == 1.0

    # TC-E2E-02: Resume same run_dir up to generation 3 without redoing gen 0..2
    orch2 = EvolutionOrchestrator(
        config=cfg,
        run_dir=run_dir,
        sandbox=sandbox,
        solve_factory=_solve_factory,
        improve_spec_factory=_improve_spec_factory,
        enforce_storage_guard=False,
    )
    cands_after_3 = orch2.run(max_generations=3)
    assert [c.generation for c in cands_after_3] == [0, 1, 2, 3]
    assert cands_after_3[3].parent_id == "cand_0002"

    # Verify CLI inspect & diff on persisted run_dir
    assert cli_main(["inspect", "--run", str(run_dir)]) == 0
    inspect_out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "cand_0002" in inspect_out
    assert "Best accepted: cand_0002" in inspect_out

    assert cli_main(["diff", "--run", str(run_dir), "cand_0000", "cand_0002"]) == 0
    diff_out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "EVOLVED_V2" in diff_out

    # Verify events log recorded cleanly
    events = [
        json.loads(line)
        for line in ArchiveStore(run_dir).events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(e["event"] == "bootstrap_complete" for e in events)
    assert sum(1 for e in events if e["event"] == "generation_complete") == 3
