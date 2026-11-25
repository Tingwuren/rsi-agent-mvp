"""Unit tests for core Pydantic v2 data schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rsi_mvp.schemas import (
    CandidateOutput,
    CandidateRecord,
    ExperimentConfig,
    FileEdit,
    LoRAAdapterRecord,
    MutationProposal,
    SuiteEvalSummary,
    TaskSolution,
    TaskSpec,
    TraceStep,
    VerifyResult,
)


def test_task_spec_to_task_input_strips_ground_truth_and_split() -> None:
    spec = TaskSpec(
        task_id="code_001",
        domain="code",
        split="holdout",
        category="algorithms",
        prompt="Write add(a, b).",
        payload={"entry_file": "solution.py"},
        ground_truth={"test_code": "assert add(1, 2) == 3"},
    )
    task_input = spec.to_task_input(num_samples=4)
    dumped = task_input.model_dump()
    assert dumped["task_id"] == "code_001"
    assert dumped["num_samples"] == 4
    assert "ground_truth" not in dumped
    assert "split" not in dumped


def test_unlabeled_task_spec_allows_none_ground_truth() -> None:
    spec = TaskSpec(
        task_id="math_unlabeled_01",
        domain="math",
        split="train",
        prompt="Compute 17 * 23.",
        ground_truth=None,
    )
    assert spec.ground_truth is None


def test_file_edit_rejects_traversal_and_bad_extensions() -> None:
    with pytest.raises(ValidationError):
        FileEdit(
            target_path="../../harness/evaluator.py",
            mode="overwrite_file",
            replace="x = 1",
        )
    with pytest.raises(ValidationError):
        FileEdit(
            target_path="/etc/passwd",
            mode="overwrite_file",
            replace="x = 1",
        )
    with pytest.raises(ValidationError):
        FileEdit(
            target_path="payload.sh",
            mode="overwrite_file",
            replace="rm -rf /",
        )
    with pytest.raises(ValidationError):
        FileEdit(
            target_path="prompts.py",
            mode="replace_block",
            search=None,
            replace="PROMPT = 'hi'",
        )


def test_mutation_proposal_roundtrip() -> None:
    proposal = MutationProposal(
        diagnosis="Missing edge-case checks on empty list input.",
        hypothesis="Add boundary analysis checklist to system prompt.",
        rationale="Most Dev failures stem from unhandled empty sequences.",
        edits=[
            FileEdit(
                target_path="prompts.py",
                mode="replace_block",
                search="SOLVER_PROMPT = 'v0'",
                replace="SOLVER_PROMPT = 'v1'",
            )
        ],
    )
    restored = MutationProposal.model_validate_json(proposal.model_dump_json())
    assert restored == proposal


def test_suite_eval_summary_rejects_passed_exceeding_total() -> None:
    with pytest.raises(ValidationError):
        SuiteEvalSummary(
            split="dev_mini",
            verifier_mode="ground_truth_code",
            mean_score=1.0,
            pass_rate=1.0,
            total_tasks=2,
            passed_tasks=3,
        )


def test_dual_track_records_roundtrip() -> None:
    lora = LoRAAdapterRecord(
        adapter_id="lora_0001",
        base_model="Qwen/Qwen3-8B",
        adapter_path="/mnt/data0/Projects/rsi-agent-mvp/.workspace/adapters/lora_0001",
        train_samples=120,
        method="sft_lora",
        train_loss=0.42,
        anchor_delta=0.01,
        accepted=True,
    )
    cand = CandidateRecord(
        candidate_id="cand_0005",
        generation=5,
        parent_id="cand_0004",
        track="weight_lora",
        lora_adapter_id=lora.adapter_id,
        status="accepted",
        snapshot_dir="snapshots/cand_0005",
        dev_mini_score=0.65,
        dev_full_score=0.62,
        anchor_score=0.72,
        holdout_score=0.60,
        output_entropy=0.48,
    )
    assert CandidateRecord.model_validate_json(cand.model_dump_json()) == cand
    assert LoRAAdapterRecord.model_validate_json(lora.model_dump_json()) == lora


def test_solution_and_verify_result_roundtrip() -> None:
    sol = TaskSolution(
        task_id="t1",
        outputs=[CandidateOutput(raw_text="42", extracted_answer="42")],
        steps=[
            TraceStep(
                step_index=0,
                action_type="llm_call",
                input_summary="prompt",
                output_summary="42",
                latency_ms=120,
                prompt_tokens=10,
                completion_tokens=2,
            )
        ],
        total_tokens=12,
        elapsed_sec=0.12,
    )
    res = VerifyResult(
        task_id="t1",
        passed=True,
        score=1.0,
        confidence=0.875,
        pseudo_label="42",
        is_label_free=True,
    )
    assert TaskSolution.model_validate_json(sol.model_dump_json()) == sol
    assert VerifyResult.model_validate_json(res.model_dump_json()) == res
    assert ExperimentConfig().hardware.serving_gpus == [1]
