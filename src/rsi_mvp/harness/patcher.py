"""Safe patch application, AST/boundary verification, Unified Diff, and dual-entry smoke tests."""

from __future__ import annotations

import ast
import difflib
import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rsi_mvp.harness.sandbox import LLMBackendSpec, SandboxExecutor
from rsi_mvp.schemas import (
    ALLOWED_EDIT_EXTENSIONS,
    MutationProposal,
    TaskSolution,
)

FailureStage = Literal["none", "validation", "apply", "ast", "boundary", "smoke"]


class PatchResult(BaseModel):
    """Result of applying and smoke-testing a ``MutationProposal`` on a candidate copy."""

    ok: bool
    candidate_dir: str
    diff_text: str = ""
    diff_path: str | None = None
    modified_files: list[str] = Field(default_factory=list)
    failure_stage: FailureStage = "none"
    error: str | None = None


def _ensure_safe_target_path(agent_root: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` inside ``agent_root``, rejecting traversal and symlink escapes."""
    root_resolved = agent_root.resolve()
    raw_target = agent_root / rel_path

    # Check every existing prefix component for symlinks escaping root
    current = agent_root
    for part in Path(rel_path).parts:
        if part in ("", ".", ".."):
            raise ValueError(f"Unsafe path component in target_path: {rel_path!r}")
        current = current / part
        if current.is_symlink():
            link_target = current.resolve()
            if not link_target.is_relative_to(root_resolved):
                raise ValueError(
                    f"Symlink escape blocked: {rel_path!r} resolves outside agent/"
                )

    resolved = raw_target.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"Path traversal blocked: {rel_path!r} escapes {root_resolved}")
    if resolved.suffix not in ALLOWED_EDIT_EXTENSIONS:
        raise ValueError(f"Disallowed file extension for edit: {rel_path!r}")
    return resolved


def check_python_syntax_and_boundary(py_path: Path) -> tuple[FailureStage, str | None]:
    """Validate Python AST syntax and enforce zero imports of ``harness`` inside ``agent/``."""
    source = py_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_path))
    except SyntaxError as exc:
        return "ast", f"SyntaxError in {py_path.name}: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "harness" in alias.name.split("."):
                    return (
                        "boundary",
                        f"Trust boundary violation in {py_path.name}: import {alias.name}",
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "harness" in module.split("."):
                return (
                    "boundary",
                    f"Trust boundary violation in {py_path.name}: from {module} import ...",
                )

    return "none", None


def apply_proposal_to_snapshot(
    parent_snapshot_dir: Path,
    candidate_snapshot_dir: Path,
    proposal: MutationProposal,
    max_files_per_mutation: int = 3,
) -> PatchResult:
    """Copy ``parent_snapshot_dir/agent`` to ``candidate_snapshot_dir/agent`` and apply edits."""
    parent_agent = parent_snapshot_dir / "agent"
    if not parent_agent.is_dir():
        return PatchResult(
            ok=False,
            candidate_dir=str(candidate_snapshot_dir),
            failure_stage="validation",
            error=f"Parent agent directory not found: {parent_agent}",
        )

    unique_targets = {e.target_path for e in proposal.edits}
    if len(unique_targets) > max_files_per_mutation:
        return PatchResult(
            ok=False,
            candidate_dir=str(candidate_snapshot_dir),
            failure_stage="validation",
            error=(
                f"Too many modified files: {len(unique_targets)} > "
                f"max_files_per_mutation ({max_files_per_mutation})"
            ),
        )

    cand_agent = candidate_snapshot_dir / "agent"
    if cand_agent.exists():
        shutil.rmtree(cand_agent)
    candidate_snapshot_dir.mkdir(parents=True, exist_ok=True)
    # Copy with symlinks=True so any injected symlink is caught explicitly by our path guard
    shutil.copytree(parent_agent, cand_agent, symlinks=True)

    before_contents: dict[str, str] = {}
    modified: list[str] = []

    for edit in proposal.edits:
        try:
            target_file = _ensure_safe_target_path(cand_agent, edit.target_path)
        except ValueError as exc:
            return PatchResult(
                ok=False,
                candidate_dir=str(candidate_snapshot_dir),
                failure_stage="validation",
                error=str(exc),
            )

        if edit.target_path not in before_contents:
            before_contents[edit.target_path] = (
                target_file.read_text(encoding="utf-8") if target_file.is_file() else ""
            )

        if edit.mode == "replace_block":
            if not target_file.is_file():
                return PatchResult(
                    ok=False,
                    candidate_dir=str(candidate_snapshot_dir),
                    failure_stage="apply",
                    error=f"replace_block target does not exist: {edit.target_path}",
                )
            current_text = target_file.read_text(encoding="utf-8")
            assert edit.search is not None
            occurrences = current_text.count(edit.search)
            if occurrences != 1:
                return PatchResult(
                    ok=False,
                    candidate_dir=str(candidate_snapshot_dir),
                    failure_stage="apply",
                    error=(
                        f"replace_block expected 1 exact match in {edit.target_path}, "
                        f"found {occurrences}"
                    ),
                )
            new_text = current_text.replace(edit.search, edit.replace, 1)
            target_file.write_text(new_text, encoding="utf-8")
        elif edit.mode == "overwrite_file":
            if not target_file.is_file():
                return PatchResult(
                    ok=False,
                    candidate_dir=str(candidate_snapshot_dir),
                    failure_stage="apply",
                    error=f"overwrite_file target does not exist: {edit.target_path}",
                )
            target_file.write_text(edit.replace, encoding="utf-8")
        elif edit.mode == "create_file":
            if target_file.exists():
                return PatchResult(
                    ok=False,
                    candidate_dir=str(candidate_snapshot_dir),
                    failure_stage="apply",
                    error=f"create_file target already exists: {edit.target_path}",
                )
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(edit.replace, encoding="utf-8")

        if edit.target_path not in modified:
            modified.append(edit.target_path)

    # Static AST & trust-boundary check on every Python file in candidate agent/
    for py_file in sorted(cand_agent.rglob("*.py")):
        stage, err = check_python_syntax_and_boundary(py_file)
        if err is not None:
            return PatchResult(
                ok=False,
                candidate_dir=str(candidate_snapshot_dir),
                modified_files=modified,
                failure_stage=stage,
                error=err,
            )

    # Generate canonical unified diff
    diff_chunks: list[str] = []
    for rel_path in modified:
        old_lines = before_contents[rel_path].splitlines(keepends=True)
        new_lines = (cand_agent / rel_path).read_text(encoding="utf-8").splitlines(keepends=True)
        diff_chunks.extend(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/agent/{rel_path}",
                tofile=f"b/agent/{rel_path}",
            )
        )
    diff_text = "".join(diff_chunks)
    diff_file = candidate_snapshot_dir / "patch.diff"
    diff_file.write_text(diff_text, encoding="utf-8")

    return PatchResult(
        ok=True,
        candidate_dir=str(candidate_snapshot_dir),
        diff_text=diff_text,
        diff_path=str(diff_file),
        modified_files=modified,
        failure_stage="none",
        error=None,
    )


def run_dual_entry_smoke_test(
    candidate_snapshot_dir: Path,
    sandbox: SandboxExecutor,
    timeout_sec: int = 15,
) -> tuple[bool, str | None]:
    """Verify both ``solve()`` and ``propose_improvement()`` contracts in an isolated sandbox."""
    # 1. Smoke-test solve()
    toy_task = {
        "task_id": "smoke_code_001",
        "domain": "code",
        "category": "smoke",
        "prompt": "Write add(a, b) returning a + b.",
        "payload": {"entry_file": "solution.py"},
        "num_samples": 1,
    }
    solve_llm = LLMBackendSpec(
        mode="mock",
        mock_responses=["```python\ndef add(a, b):\n    return a + b\n```"],
    )
    solve_res = sandbox.run_agent_solve(
        snapshot_dir=candidate_snapshot_dir,
        task_input=toy_task,
        llm_spec=solve_llm,
        timeout_sec=timeout_sec,
    )
    if not solve_res.ok or solve_res.payload is None:
        return False, f"Smoke solve() failed: {solve_res.stderr or solve_res.stdout}"
    try:
        TaskSolution.model_validate(solve_res.payload)
    except Exception as exc:
        return False, f"Smoke solve() schema validation failed: {exc}"

    # 2. Smoke-test propose_improvement() to protect recursive self-improvement capability
    dummy_proposal = {
        "diagnosis": "Smoke check diagnosis",
        "hypothesis": "Smoke check hypothesis",
        "rationale": "Smoke check rationale",
        "edits": [
            {
                "target_path": "prompts.py",
                "mode": "replace_block",
                "search": "x",
                "replace": "y",
            }
        ],
    }
    improve_ctx = {
        "parent_id": "smoke_parent",
        "dev_summary": {"pass_rate": 0.5, "total_tasks": 2},
        "failed_cases": [{"task_id": "t1", "diagnostics": "AssertionError"}],
        "source_files": {"prompts.py": "SOLVER_SYSTEM_PROMPTS = {}"},
        "recent_history": [],
    }
    improve_llm = LLMBackendSpec(
        mode="mock",
        mock_responses=[json.dumps(dummy_proposal)],
    )
    improve_res = sandbox.run_agent_improve(
        snapshot_dir=candidate_snapshot_dir,
        improve_context=improve_ctx,
        llm_spec=improve_llm,
        timeout_sec=timeout_sec,
    )
    if not improve_res.ok or improve_res.payload is None:
        return (
            False,
            f"Smoke propose_improvement() failed: {improve_res.stderr or improve_res.stdout}",
        )
    try:
        MutationProposal.model_validate(improve_res.payload)
    except Exception as exc:
        return False, f"Smoke propose_improvement() schema validation failed: {exc}"

    return True, None


def apply_and_verify_patch(
    parent_snapshot_dir: Path,
    candidate_snapshot_dir: Path,
    proposal: MutationProposal,
    sandbox: SandboxExecutor,
    max_files_per_mutation: int = 3,
    smoke_timeout_sec: int = 15,
) -> PatchResult:
    """Atomically apply a proposal to a candidate copy and run dual-entry smoke verification."""
    applied = apply_proposal_to_snapshot(
        parent_snapshot_dir=parent_snapshot_dir,
        candidate_snapshot_dir=candidate_snapshot_dir,
        proposal=proposal,
        max_files_per_mutation=max_files_per_mutation,
    )
    if not applied.ok:
        return applied

    smoke_ok, smoke_err = run_dual_entry_smoke_test(
        candidate_snapshot_dir=candidate_snapshot_dir,
        sandbox=sandbox,
        timeout_sec=smoke_timeout_sec,
    )
    if not smoke_ok:
        return applied.model_copy(
            update={"ok": False, "failure_stage": "smoke", "error": smoke_err}
        )
    return applied
