"""End-to-end M1 integration tests across both code and math domains using MockLLM."""

from __future__ import annotations

import json
from pathlib import Path

from rsi_mvp.agent import solve as agent_solve
from rsi_mvp.harness.adapters import get_adapter
from rsi_mvp.harness.data import JSONLDatasetProvider
from rsi_mvp.harness.evaluator import evaluate_suite
from rsi_mvp.harness.llm import MockLLMClient
from rsi_mvp.harness.verifiers import get_verifier
from rsi_mvp.schemas import TaskSpec


def test_m1_end_to_end_code_and_math_evaluation(tmp_path: Path) -> None:
    # 1. Code domain suite
    code_dir = tmp_path / "code_ds"
    code_dir.mkdir()
    code_tasks = [
        TaskSpec(
            task_id="c_pass",
            domain="code",
            split="dev",
            category="math",
            prompt="Write double(x).",
            payload={"entry_file": "solution.py"},
            ground_truth={
                "test_code": "from solution import double\ndef test_d(): assert double(3)==6\n"
            },
        ),
        TaskSpec(
            task_id="c_fail",
            domain="code",
            split="dev",
            category="math",
            prompt="Write triple(x).",
            payload={"entry_file": "solution.py"},
            ground_truth={
                "test_code": "from solution import triple\ndef test_t(): assert triple(3)==9\n"
            },
        ),
    ]
    (code_dir / "dev.jsonl").write_text(
        "\n".join(t.model_dump_json() for t in code_tasks) + "\n", encoding="utf-8"
    )

    def _code_handler(messages: list[dict[str, str]], _temp: float, _max: int) -> str:
        user_msg = messages[-1]["content"]
        if "double" in user_msg:
            return "```python\ndef double(x):\n    return x * 2\n```"
        return "```python\ndef triple(x):\n    return x * 2\n```"  # deliberate bug

    code_summary = evaluate_suite(
        tasks=JSONLDatasetProvider(code_dir).get_split("dev"),
        split_name="dev_full",
        solve_fn=agent_solve,
        llm_client=MockLLMClient(handler=_code_handler),
        adapter=get_adapter("code"),
        verifier=get_verifier("ground_truth_code"),
        max_concurrency=2,
        traces_dir=tmp_path / "code_traces",
    )
    assert code_summary.total_tasks == 2
    assert code_summary.passed_tasks == 1
    assert code_summary.pass_rate == 0.5
    assert code_summary.failure_breakdown == {"assertion_error": 1}
    assert (tmp_path / "code_traces" / "c_pass.json").is_file()
    saved = json.loads((tmp_path / "code_traces" / "c_fail.json").read_text(encoding="utf-8"))
    assert saved["verify"]["passed"] is False

    # 2. Math domain suite (zero harness changes — plugin swap only)
    math_dir = tmp_path / "math_ds"
    math_dir.mkdir()
    math_tasks = [
        TaskSpec(
            task_id="m_1",
            domain="math",
            split="dev",
            category="algebra",
            prompt="Compute 6 * 7.",
            ground_truth={"answer": "42"},
        ),
        TaskSpec(
            task_id="m_2",
            domain="math",
            split="dev",
            category="fractions",
            prompt="Compute 1/4 + 1/4.",
            ground_truth={"answer": r"\frac{1}{2}"},
        ),
    ]
    (math_dir / "dev.jsonl").write_text(
        "\n".join(t.model_dump_json() for t in math_tasks) + "\n", encoding="utf-8"
    )

    def _math_handler(messages: list[dict[str, str]], _temp: float, _max: int) -> str:
        user_msg = messages[-1]["content"]
        if "6 * 7" in user_msg:
            return r"6 * 7 = \boxed{42}"
        return r"1/4 + 1/4 = \boxed{0.5}"

    math_summary = evaluate_suite(
        tasks=JSONLDatasetProvider(math_dir).get_split("dev"),
        split_name="dev_full",
        solve_fn=agent_solve,
        llm_client=MockLLMClient(handler=_math_handler),
        adapter=get_adapter("math"),
        verifier=get_verifier("ground_truth_math"),
        max_concurrency=2,
        traces_dir=tmp_path / "math_traces",
    )
    assert math_summary.total_tasks == 2
    assert math_summary.passed_tasks == 2
    assert math_summary.pass_rate == 1.0
