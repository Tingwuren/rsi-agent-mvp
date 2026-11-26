"""Unit tests for mutable seed agent v0 (solve & propose_improvement)."""

from __future__ import annotations

import json

from rsi_mvp.agent import propose_improvement, solve
from rsi_mvp.harness.llm import MockLLMClient
from rsi_mvp.schemas import MutationProposal, TaskSolution


def test_agent_v0_solve_single_and_multi_sample() -> None:
    client = MockLLMClient(
        responses=[
            "<think>reasoning</think>\n```python\ndef f(x):\n    return x + 1\n```",
            "```python\ndef f(x):\n    return x + 2\n```",
        ]
    )
    raw = solve(
        {
            "task_id": "c1",
            "domain": "code",
            "category": "general",
            "prompt": "Increment x.",
            "payload": {},
            "num_samples": 2,
        },
        client,
    )
    sol = TaskSolution.model_validate(raw)
    assert len(sol.outputs) == 2
    assert "return x + 1" in sol.outputs[0].extracted_answer
    assert "return x + 2" in sol.outputs[1].extracted_answer
    assert sol.total_tokens == 160


def test_agent_v0_propose_improvement_parses_json() -> None:
    payload = {
        "diagnosis": "Prompt does not remind solver to check empty inputs.",
        "hypothesis": "Add edge-case reminder to code prompt.",
        "rationale": "Improves edge-case pass rate generally.",
        "edits": [
            {
                "target_path": "prompts.py",
                "mode": "replace_block",
                "search": "Carefully read the specification and edge cases.",
                "replace": (
                    "Carefully read the specification, empty inputs, and boundary edge cases."
                ),
            }
        ],
    }
    raw_resp = f"<think>analyze</think>\n```json\n{json.dumps(payload)}\n```"
    client = MockLLMClient(responses=[raw_resp])
    raw = propose_improvement(
        {
            "parent_id": "cand_0000",
            "dev_summary": {"pass_rate": 0.5},
            "failed_cases": [{"task_id": "t1", "diagnostics": "AssertionError"}],
            "source_files": {"prompts.py": "SOLVER_SYSTEM_PROMPTS = {}"},
            "recent_history": [],
        },
        client,
    )
    proposal = MutationProposal.model_validate(raw)
    assert proposal.edits[0].target_path == "prompts.py"
