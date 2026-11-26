"""Label-free CodeTestCoVerifier: M-code x N-generated-tests execution matrix consensus."""

from __future__ import annotations

import ast
import re

from rsi_mvp.harness.adapters.code import extract_python_code
from rsi_mvp.harness.sandbox import SandboxExecutor, SubprocessSandbox
from rsi_mvp.harness.verifiers.base import register_verifier
from rsi_mvp.schemas import TaskSolution, TaskSpec, VerifyResult

_ASSERT_LINE_RE = re.compile(r"^\s*assert\s+.+$", flags=re.MULTILINE)


def _is_degenerate_test(test_code: str) -> bool:
    """Reject syntactically invalid or trivially tautological tests (e.g. ``assert True``)."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return True

    has_meaningful_assert = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            # Reject constant-only assertions like `assert True` or `assert 1 == 1`
            if isinstance(node.test, ast.Constant):
                continue
            has_call_or_name = any(
                isinstance(sub, (ast.Call, ast.Name)) for sub in ast.walk(node.test)
            )
            if has_call_or_name:
                has_meaningful_assert = True
                break
    return not has_meaningful_assert


def _wrap_test_code(raw_test: str) -> str:
    cleaned = extract_python_code(raw_test)
    if "from solution import" not in cleaned and "import solution" not in cleaned:
        cleaned = "from solution import *\n" + cleaned
    if "def test_" not in cleaned:
        indented = "\n".join(f"    {ln}" for ln in cleaned.splitlines() if ln.strip())
        cleaned = f"from solution import *\ndef test_generated():\n{indented}\n"
    return cleaned


class CodeTestCoVerifier:
    """Label-free code verifier using M candidate codes x N generated tests (CodeT style).

    Builds an isolated execution pass matrix ``A[i][j]`` across candidate implementations
    and agent-generated unit tests, filters degenerate/tautological tests, and scores each
    implementation by dual consensus across informative tests.
    """

    mode_name = "code_test_co"
    is_label_free = True

    def __init__(
        self,
        sandbox: SandboxExecutor | None = None,
        min_consensus: float = 0.5,
    ) -> None:
        self._sandbox: SandboxExecutor = sandbox or SubprocessSandbox()
        self.min_consensus = min_consensus

    def _collect_tests(self, spec: TaskSpec, solution: TaskSolution) -> list[str]:
        tests: list[str] = []
        for out in solution.outputs:
            gen = out.auxiliary.get("generated_tests")
            if isinstance(gen, str) and gen.strip():
                tests.append(_wrap_test_code(gen))
            elif isinstance(gen, list):
                for item in gen:
                    if isinstance(item, str) and item.strip():
                        tests.append(_wrap_test_code(item))

        # Fallback: extract public example assertions embedded in the prompt if no auxiliary tests
        if not tests:
            prompt_asserts = _ASSERT_LINE_RE.findall(spec.prompt)
            if prompt_asserts:
                body = "\n".join(f"    {a.strip()}" for a in prompt_asserts)
                tests.append(f"from solution import *\ndef test_prompt_examples():\n{body}\n")

        return [t for t in tests if not _is_degenerate_test(t)]

    def verify(self, spec: TaskSpec, solution: TaskSolution) -> VerifyResult:
        if solution.error:
            return VerifyResult(
                task_id=spec.task_id,
                passed=False,
                score=0.0,
                confidence=0.0,
                pseudo_label=None,
                is_label_free=True,
                failure_type="agent_crash",
                diagnostics=solution.error[:1024],
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec,
            )

        entry_file = str(spec.payload.get("entry_file", "solution.py"))
        raw_codes = [
            extract_python_code(out.extracted_answer or out.raw_text)
            for out in solution.outputs
        ]
        # Keep syntactically valid implementations only
        valid_codes: list[str] = []
        for code in raw_codes:
            if not code.strip():
                continue
            try:
                compile(code, entry_file, "exec")
                valid_codes.append(code)
            except SyntaxError:
                continue

        if not valid_codes:
            return VerifyResult(
                task_id=spec.task_id,
                passed=False,
                score=0.0,
                confidence=0.0,
                pseudo_label=None,
                is_label_free=True,
                failure_type="syntax_error",
                diagnostics="All sampled implementations failed Python syntax compilation.",
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec,
            )

        valid_tests = self._collect_tests(spec, solution)
        if not valid_tests:
            return VerifyResult(
                task_id=spec.task_id,
                passed=False,
                score=0.0,
                confidence=0.0,
                pseudo_label=None,
                is_label_free=True,
                failure_type="low_consensus",
                diagnostics=(
                    "No valid non-degenerate generated tests available for co-verification."
                ),
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec,
            )

        # Build M x N execution matrix A[i][j] in {0, 1}
        m = len(valid_codes)
        n = len(valid_tests)
        matrix: list[list[int]] = [[0] * n for _ in range(m)]
        eval_elapsed = 0.0
        per_cell_timeout = max(2, min(spec.timeout_sec, 10))

        for i, code in enumerate(valid_codes):
            for j, test_code in enumerate(valid_tests):
                res = self._sandbox.run_pytest_eval(
                    solution_code=code,
                    entry_file=entry_file,
                    test_code=test_code,
                    timeout_sec=per_cell_timeout,
                )
                eval_elapsed += res.elapsed_sec
                matrix[i][j] = 1 if (res.ok and not res.tampered) else 0

        # Filter out dead tests (0 codes pass)
        col_sums = [sum(matrix[i][j] for i in range(m)) for j in range(n)]
        active_cols = [j for j in range(n) if col_sums[j] > 0]
        if not active_cols:
            return VerifyResult(
                task_id=spec.task_id,
                passed=False,
                score=0.0,
                confidence=0.0,
                pseudo_label=None,
                is_label_free=True,
                failure_type="low_consensus",
                diagnostics=f"All {n} candidate tests failed on all {m} implementations.",
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec + eval_elapsed,
            )

        # CodeT-inspired dual consensus score: reward codes passing tests agreed upon by peers
        max_possible_weight = sum(col_sums[j] for j in active_cols)
        best_idx = 0
        best_dual_score = -1.0
        best_pass_frac = 0.0
        for i in range(m):
            weighted_hits = sum(matrix[i][j] * col_sums[j] for j in active_cols)
            dual_score = weighted_hits / max_possible_weight
            pass_frac = sum(matrix[i][j] for j in active_cols) / len(active_cols)
            if dual_score > best_dual_score:
                best_dual_score = dual_score
                best_pass_frac = pass_frac
                best_idx = i

        confidence = round(best_pass_frac, 4)
        score = round(best_dual_score, 4)
        passed = confidence >= (self.min_consensus - 1e-9)
        diag = (
            f"matrix={m}x{n} active_tests={len(active_cols)} "
            f"best_code_idx={best_idx} pass_frac={confidence:.2f} dual_score={score:.2f}"
        )
        return VerifyResult(
            task_id=spec.task_id,
            passed=passed,
            score=score if passed else 0.0,
            confidence=confidence,
            pseudo_label=valid_codes[best_idx] if passed else None,
            is_label_free=True,
            failure_type="none" if passed else "low_consensus",
            diagnostics=diag,
            tokens_used=solution.total_tokens,
            elapsed_sec=solution.elapsed_sec + eval_elapsed,
        )


register_verifier("code_test_co", CodeTestCoVerifier)
