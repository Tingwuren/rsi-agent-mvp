"""Unit tests for SubprocessSandbox, DockerSandbox command builder, and worker isolation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from rsi_mvp.harness.adapters import get_adapter
from rsi_mvp.harness.evaluator import evaluate_suite, make_sandboxed_solve_fn
from rsi_mvp.harness.llm import MockLLMClient
from rsi_mvp.harness.sandbox import (
    DockerSandbox,
    LLMBackendSpec,
    SubprocessSandbox,
    create_sandbox,
)
from rsi_mvp.harness.verifiers import GroundTruthCodeVerifier
from rsi_mvp.schemas import MutationProposal, SandboxConfig, TaskSolution, TaskSpec

_SEED_AGENT_DIR = Path("src/rsi_mvp/agent")


def _copy_seed_snapshot(dest: Path) -> Path:
    snap = dest / "cand_0000"
    shutil.copytree(_SEED_AGENT_DIR, snap / "agent")
    return snap


def test_subprocess_sandbox_solve_and_improve_isolated(tmp_path: Path) -> None:
    snap = _copy_seed_snapshot(tmp_path)
    sandbox = SubprocessSandbox(tmp_root=tmp_path / "tmp")

    # 1. Verify solve() works cleanly inside isolated worker
    solve_res = sandbox.run_agent_solve(
        snapshot_dir=snap,
        task_input={
            "task_id": "sbx_01",
            "domain": "code",
            "category": "test",
            "prompt": "Write f(x) returning x + 1.",
            "payload": {},
            "num_samples": 1,
        },
        llm_spec=LLMBackendSpec(
            mode="mock",
            mock_responses=["```python\ndef f(x):\n    return x + 1\n```"],
        ),
        timeout_sec=15,
    )
    assert solve_res.ok is True, solve_res.stderr
    assert solve_res.payload is not None
    sol = TaskSolution.model_validate(solve_res.payload)
    assert "return x + 1" in sol.outputs[0].extracted_answer

    # 2. Verify propose_improvement() works cleanly inside isolated worker
    proposal_json = json.dumps(
        {
            "diagnosis": "d",
            "hypothesis": "h",
            "rationale": "r",
            "edits": [
                {
                    "target_path": "prompts.py",
                    "mode": "replace_block",
                    "search": "a",
                    "replace": "b",
                }
            ],
        }
    )
    imp_res = sandbox.run_agent_improve(
        snapshot_dir=snap,
        improve_context={
            "parent_id": "cand_0000",
            "dev_summary": {"pass_rate": 0.5},
            "failed_cases": [],
            "source_files": {"prompts.py": "a"},
            "recent_history": [],
        },
        llm_spec=LLMBackendSpec(mode="mock", mock_responses=[proposal_json]),
        timeout_sec=15,
    )
    assert imp_res.ok is True, imp_res.stderr
    assert imp_res.payload is not None
    MutationProposal.model_validate(imp_res.payload)


def test_sandbox_immune_to_stdout_noise_and_loads_mutated_snapshot(tmp_path: Path) -> None:
    snap = _copy_seed_snapshot(tmp_path)
    # Inject noisy stdout AND a mutated prompt marker directly into snapshot's agent/
    prompts_file = snap / "agent" / "prompts.py"
    prompts_file.write_text(
        prompts_file.read_text(encoding="utf-8") + "\nMUTATION_MARKER = 'MUTATED_V99'\n",
        encoding="utf-8",
    )
    core_file = snap / "agent" / "core.py"
    core_file.write_text(
        "from rsi_mvp.agent.prompts import MUTATION_MARKER\n"
        "def solve(task, llm_client):\n"
        "    print('GARBAGE STDOUT {{{ not json !!!')\n"
        "    return {\n"
        "        'task_id': task['task_id'],\n"
        "        'outputs': [{'raw_text': MUTATION_MARKER, 'extracted_answer': MUTATION_MARKER}],\n"
        "        'steps': [],\n"
        "        'error': None,\n"
        "        'total_tokens': 0,\n"
        "        'elapsed_sec': 0.0,\n"
        "    }\n",
        encoding="utf-8",
    )

    sandbox = SubprocessSandbox(tmp_root=tmp_path / "tmp")
    res = sandbox.run_agent_solve(
        snapshot_dir=snap,
        task_input={"task_id": "mut_check", "domain": "code", "prompt": "hi"},
        llm_spec=LLMBackendSpec(mode="mock", mock_responses=[]),
        timeout_sec=10,
    )
    assert res.ok is True, res.stderr
    assert res.payload is not None
    assert res.payload["outputs"][0]["extracted_answer"] == "MUTATED_V99"
    assert "GARBAGE STDOUT" in res.stdout


def test_docker_sandbox_command_hardening_and_factory(tmp_path: Path) -> None:
    dkr = DockerSandbox(image="python:3.11-slim", memory_limit="1g", cpus="2", pids_limit=128)
    cmd = dkr.build_docker_command(
        container_name="rsi-test-1",
        mounts=[(tmp_path, "/work", "ro")],
        inner_cmd=["python3", "-c", "print(1)"],
        network_mode="none",
    )
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "--read-only" in cmd
    assert "--cap-drop" in cmd and cmd[cmd.index("--cap-drop") + 1] == "ALL"
    assert "--pids-limit" in cmd and cmd[cmd.index("--pids-limit") + 1] == "128"
    assert "--user" in cmd and cmd[cmd.index("--user") + 1] == "65534:65534"
    assert any(arg.endswith(":/work:ro") for arg in cmd)

    assert isinstance(create_sandbox(SandboxConfig(backend="subprocess")), SubprocessSandbox)
    assert isinstance(create_sandbox(SandboxConfig(backend="docker")), DockerSandbox)


def test_evaluator_runs_untrusted_snapshot_inside_sandbox(tmp_path: Path) -> None:
    snap = _copy_seed_snapshot(tmp_path)
    sandbox = SubprocessSandbox(tmp_root=tmp_path / "tmp")
    solve_fn = make_sandboxed_solve_fn(
        snapshot_dir=snap,
        sandbox=sandbox,
        llm_spec=LLMBackendSpec(
            mode="mock",
            mock_responses=["```python\ndef inc(x):\n    return x + 1\n```"],
        ),
        timeout_sec=15,
    )
    tasks = [
        TaskSpec(
            task_id="sbx_eval_1",
            domain="code",
            split="dev",
            prompt="Write inc(x).",
            payload={"entry_file": "solution.py"},
            ground_truth={
                "test_code": "from solution import inc\ndef test_i(): assert inc(4)==5\n"
            },
        )
    ]
    summary = evaluate_suite(
        tasks=tasks,
        split_name="dev_mini",
        solve_fn=solve_fn,
        llm_client=MockLLMClient(),
        adapter=get_adapter("code"),
        verifier=GroundTruthCodeVerifier(sandbox=sandbox),
        max_concurrency=1,
        traces_dir=tmp_path / "traces",
    )
    assert summary.passed_tasks == 1
    assert summary.pass_rate == 1.0
    assert (tmp_path / "traces" / "sbx_eval_1.json").is_file()

    # Now mutate snapshot to hang infinitely and ensure evaluator catches it without crashing
    (snap / "agent" / "core.py").write_text(
        "def solve(task, llm_client):\n    while True:\n        pass\n",
        encoding="utf-8",
    )
    bad_solve_fn = make_sandboxed_solve_fn(
        snapshot_dir=snap,
        sandbox=sandbox,
        llm_spec=LLMBackendSpec(mode="mock", mock_responses=[]),
        timeout_sec=2,
    )
    bad_summary = evaluate_suite(
        tasks=tasks,
        split_name="dev_mini",
        solve_fn=bad_solve_fn,
        llm_client=MockLLMClient(),
        adapter=get_adapter("code"),
        verifier=GroundTruthCodeVerifier(sandbox=sandbox),
        max_concurrency=1,
    )
    assert bad_summary.passed_tasks == 0
    assert bad_summary.failure_breakdown == {"agent_crash": 1}
    assert "timeout" in bad_summary.results[0].diagnostics.lower()
