"""End-to-end M5 integration test: Label-free consensus RSI + ground-truth Anchor/Holdout."""

from __future__ import annotations

import json
from pathlib import Path

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


def _write_unlabeled_math_dataset(ds_dir: Path) -> None:
    ds_dir.mkdir(parents=True, exist_ok=True)
    # Dev split is strictly UNLABELED (ground_truth=None)
    dev_tasks = [
        TaskSpec(
            task_id="lf_dev_1",
            domain="math",
            split="dev",
            category="algebra",
            prompt="Compute 3 * 7.",
            ground_truth=None,
        ),
        TaskSpec(
            task_id="lf_dev_2",
            domain="math",
            split="dev",
            category="fractions",
            prompt="Simplify 2/4.",
            ground_truth=None,
        ),
    ]
    # Anchor & Holdout retain hidden golden labels for true generalization & anti-collapse checks
    anchor_tasks = [
        TaskSpec(
            task_id="anc_m1",
            domain="math",
            split="anchor",
            category="algebra",
            prompt="Compute 5 + 5.",
            ground_truth={"answer": "10"},
        )
    ]
    holdout_tasks = [
        TaskSpec(
            task_id="ho_m1",
            domain="math",
            split="holdout",
            category="algebra",
            prompt="Compute 9 * 9.",
            ground_truth={"answer": "81"},
        )
    ]
    for name, items in [("dev", dev_tasks), ("anchor", anchor_tasks), ("holdout", holdout_tasks)]:
        (ds_dir / f"{name}.jsonl").write_text(
            "\n".join(t.model_dump_json() for t in items) + "\n", encoding="utf-8"
        )


def test_m5_end_to_end_labelfree_consensus_with_golden_anchor_and_report(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds_lf"
    _write_unlabeled_math_dataset(ds_dir)
    ws = tmp_path / "ws"
    (ws / "tmp").mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        experiment=ExperimentMetaConfig(
            name="m5_labelfree_math",
            seed=42,
            workspace_root=str(ws),
            output_dir=str(ws / "runs"),
        ),
        task=TaskConfig(adapter="math", verifier="consensus", num_samples_per_task=4),
        dataset=DatasetConfig(
            provider="jsonl",
            path=str(ds_dir),
            mini_eval_size=2,
            full_eval_top_k=2,
            max_concurrency=1,
        ),
        evolution=EvolutionConfig(
            max_generations=1,
            selection_policy="hill_climbing",
            holdout_eval_interval=1,
        ),
        guardrails=GuardrailsConfig(anchor_enabled=True, max_anchor_drop=0.05),
        sandbox=SandboxConfig(backend="subprocess", task_timeout_sec=15, improve_timeout_sec=20),
    )

    sandbox = SubprocessSandbox(tmp_root=ws / "tmp")

    def _solve_factory(_gen: int, snap_dir: Path) -> tuple[SolveFn, BaseLLMClient]:
        prompts_src = (snap_dir / "agent" / "prompts.py").read_text(encoding="utf-8")
        improved = "CONSENSUS_V1" in prompts_src

        def _dyn_solve(task_input: dict[str, object], _client: BaseLLMClient) -> dict[str, object]:
            tid = str(task_input["task_id"])
            if tid == "lf_dev_1":
                resps = [r"\boxed{21}", r"\boxed{21}", r"\boxed{21}", r"\boxed{20}"]
            elif tid == "lf_dev_2":
                # v0 has low consensus (4 different answers); evolved agent converges on 1/2!
                resps = (
                    [r"\boxed{\frac{1}{2}}", r"\boxed{0.5}", r"\boxed{1/2}", r"\boxed{0.5}"]
                    if improved
                    else [r"\boxed{1}", r"\boxed{2}", r"\boxed{3}", r"\boxed{4}"]
                )
            elif tid == "anc_m1":
                resps = [r"\boxed{10}"]
            else:
                resps = [r"\boxed{81}"] if improved else [r"\boxed{72}"]

            return make_sandboxed_solve_fn(
                snapshot_dir=snap_dir,
                sandbox=sandbox,
                llm_spec=LLMBackendSpec(mode="mock", mock_responses=resps),
                timeout_sec=15,
            )(task_input, _client)

        return _dyn_solve, MockLLMClient()

    def _improve_factory(_gen: int, _parent: object) -> LLMBackendSpec:
        prop = {
            "diagnosis": "Math solver outputs inconsistent fractions across samples.",
            "hypothesis": "Add CONSENSUS_V1 step-by-step verification reminder.",
            "rationale": "Improves self-consistency and true accuracy.",
            "edits": [
                {
                    "target_path": "prompts.py",
                    "mode": "replace_block",
                    "search": "Solve the problem step by step",
                    "replace": "CONSENSUS_V1: Solve the problem step by step",
                }
            ],
        }
        return LLMBackendSpec(mode="mock", mock_responses=[json.dumps(prop)])

    run_dir = ws / "runs" / "m5_lf_run"
    orch = EvolutionOrchestrator(
        config=cfg,
        run_dir=run_dir,
        sandbox=sandbox,
        solve_factory=_solve_factory,
        improve_spec_factory=_improve_factory,
        enforce_storage_guard=False,
    )
    cands = orch.run(max_generations=1)

    assert len(cands) == 2
    # v0: 1/2 unlabeled dev consensus, 1.0 real anchor, 0.0 real holdout
    assert cands[0].dev_full_score == 0.5
    assert cands[0].anchor_score == 1.0
    assert cands[0].holdout_score == 0.0
    assert cands[0].output_entropy is not None and cands[0].output_entropy > 0.0

    # Gen 1: 1.0 unlabeled dev consensus, 1.0 real anchor, 1.0 real holdout!
    assert cands[1].status == "accepted"
    assert cands[1].dev_full_score == 1.0
    assert cands[1].anchor_score == 1.0
    assert cands[1].holdout_score == 1.0

    # Automatic summary.md generated at end of run
    summary_path = run_dir / "summary.md"
    assert summary_path.is_file()
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "CONSENSUS_V1" in summary_text
    assert "Holdout Score" in summary_text
