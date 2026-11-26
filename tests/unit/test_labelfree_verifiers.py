"""Unit tests for label-free verifiers (TC-LF-01 & TC-LF-02)."""

from __future__ import annotations

from pathlib import Path

from rsi_mvp.harness.sandbox import SubprocessSandbox
from rsi_mvp.harness.verifiers import CodeTestCoVerifier, ConsensusVerifier, get_verifier
from rsi_mvp.schemas import CandidateOutput, TaskSolution, TaskSpec


def test_tc_lf_01_consensus_verifier_math_symbolic_and_low_consensus() -> None:
    verifier = get_verifier("consensus")
    assert isinstance(verifier, ConsensusVerifier)
    assert verifier.is_label_free is True

    spec = TaskSpec(
        task_id="lf_math_01",
        domain="math",
        split="dev",
        prompt="Simplify 2/4.",
        ground_truth=None,  # Strictly unlabeled!
    )

    # 6/8 symbolically equivalent answers (1/2, 0.5, \frac{2}{4}) -> high consensus (0.75)
    high_sol = TaskSolution(
        task_id="lf_math_01",
        outputs=[
            CandidateOutput(raw_text="", extracted_answer=r"\frac{1}{2}"),
            CandidateOutput(raw_text="", extracted_answer="0.5"),
            CandidateOutput(raw_text="", extracted_answer=r"\frac{2}{4}"),
            CandidateOutput(raw_text="", extracted_answer="1/2"),
            CandidateOutput(raw_text="", extracted_answer="0.5"),
            CandidateOutput(raw_text="", extracted_answer="0.5"),
            CandidateOutput(raw_text="", extracted_answer="2"),
            CandidateOutput(raw_text="", extracted_answer="1"),
        ],
    )
    res_high = verifier.verify(spec, high_sol)
    assert res_high.passed is True
    assert res_high.confidence == 0.75
    assert res_high.pseudo_label is not None
    assert res_high.failure_type == "none"

    # Scattered answers (6 distinct answers out of 8) -> low_consensus rejected!
    low_sol = TaskSolution(
        task_id="lf_math_01",
        outputs=[
            CandidateOutput(raw_text="", extracted_answer=str(v))
            for v in [1, 2, 3, 4, 5, 6, 1, 2]
        ],
    )
    res_low = verifier.verify(spec, low_sol)
    assert res_low.passed is False
    assert res_low.confidence == 0.25
    assert res_low.pseudo_label is None
    assert res_low.failure_type == "low_consensus"


def test_tc_lf_02_code_test_co_verifier_matrix_and_degenerate_filter(tmp_path: Path) -> None:
    sandbox = SubprocessSandbox(tmp_root=tmp_path / "tmp")
    verifier = CodeTestCoVerifier(sandbox=sandbox, min_consensus=0.5)
    assert verifier.is_label_free is True

    spec = TaskSpec(
        task_id="lf_code_01",
        domain="code",
        split="dev",
        prompt="Write a Python function square(x) returning x * x.",
        payload={"entry_file": "solution.py"},
        ground_truth=None,  # Strictly unlabeled!
        timeout_sec=10,
    )

    good_code_1 = "def square(x):\n    return x * x\n"
    good_code_2 = "def square(x):\n    return x ** 2\n"
    buggy_code = "def square(x):\n    return x + x\n"

    real_test_1 = "from solution import square\ndef test_pos():\n    assert square(3) == 9\n"
    real_test_2 = "from solution import square\ndef test_neg():\n    assert square(-4) == 16\n"
    tautology_test = "def test_fake():\n    assert True\n"

    sol = TaskSolution(
        task_id="lf_code_01",
        outputs=[
            CandidateOutput(
                raw_text="",
                extracted_answer=buggy_code,
                auxiliary={"generated_tests": tautology_test},
            ),
            CandidateOutput(
                raw_text="",
                extracted_answer=good_code_1,
                auxiliary={"generated_tests": real_test_1},
            ),
            CandidateOutput(
                raw_text="",
                extracted_answer=good_code_2,
                auxiliary={"generated_tests": real_test_2},
            ),
        ],
    )
    res = verifier.verify(spec, sol)
    assert res.passed is True, res.diagnostics
    assert res.confidence == 1.0
    assert res.pseudo_label in (good_code_1.strip(), good_code_2.strip())
    assert "active_tests=2" in res.diagnostics
