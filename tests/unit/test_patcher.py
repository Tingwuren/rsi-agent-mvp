"""Unit tests for safe Patcher (TC-PATCH-01 ~ TC-PATCH-03 and dual-entry smoke tests)."""

from __future__ import annotations

import shutil
from pathlib import Path

from rsi_mvp.harness.patcher import apply_and_verify_patch, apply_proposal_to_snapshot
from rsi_mvp.harness.sandbox import SubprocessSandbox
from rsi_mvp.schemas import FileEdit, MutationProposal

_SEED_AGENT_DIR = Path("src/rsi_mvp/agent")


def _make_parent_snapshot(tmp_path: Path) -> Path:
    parent = tmp_path / "cand_0000"
    shutil.copytree(_SEED_AGENT_DIR, parent / "agent")
    return parent


def test_tc_patch_01_replace_block_and_full_smoke_pass(tmp_path: Path) -> None:
    parent = _make_parent_snapshot(tmp_path)
    original_prompts = (parent / "agent" / "prompts.py").read_text(encoding="utf-8")

    proposal = MutationProposal(
        diagnosis="Solver prompt can emphasize boundary conditions.",
        hypothesis="Remind solver to double-check empty and negative inputs.",
        rationale="General improvement across Python coding tasks.",
        edits=[
            FileEdit(
                target_path="agent/prompts.py",
                mode="replace_block",
                search="Carefully read the specification and edge cases.",
                replace="Carefully read the specification, empty inputs, and boundary edge cases.",
            )
        ],
    )
    candidate = tmp_path / "cand_0001"
    sandbox = SubprocessSandbox(tmp_root=tmp_path / "tmp")
    res = apply_and_verify_patch(
        parent_snapshot_dir=parent,
        candidate_snapshot_dir=candidate,
        proposal=proposal,
        sandbox=sandbox,
        smoke_timeout_sec=15,
    )
    assert res.ok is True, res.error
    assert res.failure_stage == "none"
    assert res.modified_files == ["prompts.py"]
    assert res.diff_path is not None and Path(res.diff_path).is_file()
    assert "+        \"Carefully read the specification, empty inputs" in res.diff_text
    # Parent snapshot MUST remain completely untouched
    assert (parent / "agent" / "prompts.py").read_text(encoding="utf-8") == original_prompts


def test_tc_patch_02_missing_or_ambiguous_search_rejected(tmp_path: Path) -> None:
    parent = _make_parent_snapshot(tmp_path)

    # 0 matches
    p_missing = MutationProposal(
        diagnosis="d",
        hypothesis="h",
        rationale="r",
        edits=[
            FileEdit(
                target_path="prompts.py",
                mode="replace_block",
                search="NON_EXISTENT_ANCHOR_STRING",
                replace="NEW",
            )
        ],
    )
    r1 = apply_proposal_to_snapshot(parent, tmp_path / "cand_missing", p_missing)
    assert r1.ok is False
    assert r1.failure_stage == "apply"
    assert "found 0" in (r1.error or "")

    # Multiple matches ("return")
    p_ambiguous = MutationProposal(
        diagnosis="d",
        hypothesis="h",
        rationale="r",
        edits=[
            FileEdit(
                target_path="tools.py",
                mode="replace_block",
                search="return ",
                replace="return  ",
            )
        ],
    )
    r2 = apply_proposal_to_snapshot(parent, tmp_path / "cand_multi", p_ambiguous)
    assert r2.ok is False
    assert r2.failure_stage == "apply"


def test_tc_patch_03_syntax_error_and_file_limit_rejected(tmp_path: Path) -> None:
    parent = _make_parent_snapshot(tmp_path)

    p_syntax = MutationProposal(
        diagnosis="d",
        hypothesis="h",
        rationale="r",
        edits=[
            FileEdit(
                target_path="tools.py",
                mode="overwrite_file",
                replace="def broken(:\n    pass\n",
            )
        ],
    )
    r_syn = apply_proposal_to_snapshot(parent, tmp_path / "cand_syn", p_syntax)
    assert r_syn.ok is False
    assert r_syn.failure_stage == "ast"

    # Too many modified files (> max_files_per_mutation=1)
    p_many = MutationProposal(
        diagnosis="d",
        hypothesis="h",
        rationale="r",
        edits=[
            FileEdit(target_path="helper1.py", mode="create_file", replace="X = 1\n"),
            FileEdit(target_path="helper2.py", mode="create_file", replace="Y = 2\n"),
        ],
    )
    r_many = apply_proposal_to_snapshot(
        parent, tmp_path / "cand_many", p_many, max_files_per_mutation=1
    )
    assert r_many.ok is False
    assert r_many.failure_stage == "validation"
