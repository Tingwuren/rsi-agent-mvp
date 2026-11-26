"""Unit tests for deterministic GroundTruthCodeVerifier and GroundTruthMathVerifier."""

from __future__ import annotations

from rsi_mvp.harness.verifiers import get_verifier, math_answers_equivalent
from rsi_mvp.schemas import CandidateOutput, TaskSolution, TaskSpec


def test_code_verifier_pass_fail_syntax_and_timeout() -> None:
    verifier = get_verifier("ground_truth_code")
    spec = TaskSpec(
        task_id="add_task",
        domain="code",
        split="dev",
        prompt="Write add(a, b).",
        payload={"entry_file": "solution.py"},
        ground_truth={
            "test_code": (
                "from solution import add\n"
                "def test_one():\n    assert add(1, 2) == 3\n"
                "def test_two():\n    assert add(-1, 5) == 4\n"
            )
        },
        timeout_sec=3,
    )

    good_sol = TaskSolution(
        task_id="add_task",
        outputs=[
            CandidateOutput(raw_text="", extracted_answer="def add(a, b):\n    return a + b\n")
        ],
    )
    res_good = verifier.verify(spec, good_sol)
    assert res_good.passed is True
    assert res_good.score == 1.0
    assert res_good.failure_type == "none"

    partial_sol = TaskSolution(
        task_id="add_task",
        outputs=[CandidateOutput(raw_text="", extracted_answer="def add(a, b):\n    return 3\n")],
    )
    res_partial = verifier.verify(spec, partial_sol)
    assert res_partial.passed is False
    assert res_partial.score == 0.5
    assert res_partial.failure_type == "assertion_error"

    syntax_sol = TaskSolution(
        task_id="add_task",
        outputs=[
            CandidateOutput(raw_text="", extracted_answer="def add(a, b)\n    return a + b\n")
        ],
    )
    res_syn = verifier.verify(spec, syntax_sol)
    assert res_syn.passed is False
    assert res_syn.failure_type == "syntax_error"

    timeout_sol = TaskSolution(
        task_id="add_task",
        outputs=[
            CandidateOutput(
                raw_text="",
                extracted_answer="def add(a, b):\n    while True:\n        pass\n",
            )
        ],
    )
    res_to = verifier.verify(spec, timeout_sol)
    assert res_to.passed is False
    assert res_to.failure_type == "timeout"


def test_math_verifier_symbolic_equivalence() -> None:
    assert math_answers_equivalent(r"\frac{1}{2}", "0.5")
    assert math_answers_equivalent(r"2\pi", "2 * pi")
    assert math_answers_equivalent("(x+1)**2", "x^2 + 2*x + 1")
    assert math_answers_equivalent("58500", "58,500")
    assert not math_answers_equivalent("(3, 5)", "(35)")
    assert not math_answers_equivalent("41", "42")

    verifier = get_verifier("ground_truth_math")
    spec = TaskSpec(
        task_id="m1",
        domain="math",
        split="dev",
        prompt="Simplify 2/4.",
        ground_truth={"answer": r"\frac{1}{2}"},
    )
    sol = TaskSolution(
        task_id="m1",
        outputs=[CandidateOutput(raw_text=r"Answer is \boxed{0.5}", extracted_answer="0.5")],
    )
    res = verifier.verify(spec, sol)
    assert res.passed is True
    assert res.score == 1.0
