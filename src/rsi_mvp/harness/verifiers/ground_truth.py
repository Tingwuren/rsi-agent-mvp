"""Deterministic ground-truth verifiers for Python code (pytest) and symbolic math."""

from __future__ import annotations

import re

from rsi_mvp.harness.adapters.code import extract_python_code
from rsi_mvp.harness.adapters.math import extract_boxed_answer
from rsi_mvp.harness.sandbox import SandboxExecutor, SubprocessSandbox
from rsi_mvp.harness.verifiers.base import register_verifier
from rsi_mvp.schemas import FailureType, TaskSolution, TaskSpec, VerifyResult

_MAX_DIAG_CHARS = 4096
_PASSED_RE = re.compile(r"(\d+)\s+passed")
_FAILED_RE = re.compile(r"(\d+)\s+failed")
_ERROR_RE = re.compile(r"(\d+)\s+error")


def _truncate(text: str, limit: int = _MAX_DIAG_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2 - 20
    return f"{text[:half]}\n...[truncated]...\n{text[-half:]}"


_RUNTIME_ERR_RE = re.compile(
    r"^\s*E\s+(?:NameError|TypeError|ValueError|ZeroDivisionError|IndexError|"
    r"KeyError|AttributeError|RecursionError|ImportError|ModuleNotFoundError|"
    r"RuntimeError|MemoryError|OverflowError)\b",
    flags=re.MULTILINE,
)
_ASSERT_ERR_RE = re.compile(r"^\s*E\s+(?:AssertionError\b|assert\s)", flags=re.MULTILINE)


def _classify_pytest_failure(output: str, timed_out: bool) -> FailureType:
    if timed_out:
        return "timeout"
    if "SyntaxError" in output or "IndentationError" in output:
        return "syntax_error"
    if _RUNTIME_ERR_RE.search(output):
        return "runtime_error"
    if _ASSERT_ERR_RE.search(output) or "AssertionError" in output:
        return "assertion_error"
    return "runtime_error"


class GroundTruthCodeVerifier:
    """Evaluates generated Python code against immutable pytest tests inside a sandbox."""

    mode_name = "ground_truth_code"
    is_label_free = False

    def __init__(self, sandbox: SandboxExecutor | None = None) -> None:
        self._sandbox: SandboxExecutor = sandbox or SubprocessSandbox()

    def verify(self, spec: TaskSpec, solution: TaskSolution) -> VerifyResult:
        if solution.error:
            return VerifyResult(
                task_id=spec.task_id,
                passed=False,
                score=0.0,
                confidence=1.0,
                is_label_free=False,
                failure_type="agent_crash",
                diagnostics=_truncate(solution.error),
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec,
            )

        if not spec.ground_truth or "test_code" not in spec.ground_truth:
            raise ValueError(f"Task {spec.task_id} missing ground_truth['test_code']")

        raw_code = solution.outputs[0].extracted_answer or solution.outputs[0].raw_text
        code = extract_python_code(raw_code)
        entry_file = str(spec.payload.get("entry_file", "solution.py"))
        test_code = str(spec.ground_truth["test_code"])

        # Pre-check syntax to give fast, clean diagnostics
        try:
            compile(code, entry_file, "exec")
        except SyntaxError as exc:
            return VerifyResult(
                task_id=spec.task_id,
                passed=False,
                score=0.0,
                confidence=1.0,
                is_label_free=False,
                failure_type="syntax_error",
                diagnostics=_truncate(f"SyntaxError: {exc}"),
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec,
            )

        sbx_res = self._sandbox.run_pytest_eval(
            solution_code=code,
            entry_file=entry_file,
            test_code=test_code,
            timeout_sec=spec.timeout_sec,
        )
        combined = (sbx_res.stdout + "\n" + sbx_res.stderr).strip()

        if sbx_res.ok and not sbx_res.tampered:
            return VerifyResult(
                task_id=spec.task_id,
                passed=True,
                score=1.0,
                confidence=1.0,
                is_label_free=False,
                failure_type="none",
                diagnostics=_truncate(combined),
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec + sbx_res.elapsed_sec,
            )

        if sbx_res.tampered:
            return VerifyResult(
                task_id=spec.task_id,
                passed=False,
                score=0.0,
                confidence=1.0,
                is_label_free=False,
                failure_type="runtime_error",
                diagnostics=_truncate(combined),
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec + sbx_res.elapsed_sec,
            )

        # Compute partial score if pytest reported passed/failed counts
        passed_m = _PASSED_RE.search(combined)
        failed_m = _FAILED_RE.search(combined)
        error_m = _ERROR_RE.search(combined)
        n_pass = int(passed_m.group(1)) if passed_m else 0
        n_fail = int(failed_m.group(1)) if failed_m else 0
        n_err = int(error_m.group(1)) if error_m else 0
        total = n_pass + n_fail + n_err
        partial = (n_pass / total) if total > 0 else 0.0

        return VerifyResult(
            task_id=spec.task_id,
            passed=False,
            score=round(partial, 4),
            confidence=1.0,
            is_label_free=False,
            failure_type=_classify_pytest_failure(combined, timed_out=sbx_res.timed_out),
            diagnostics=_truncate(combined),
            tokens_used=solution.total_tokens,
            elapsed_sec=solution.elapsed_sec + sbx_res.elapsed_sec,
        )


def _normalize_latex_math(expr: str) -> str:
    """Clean common LaTeX wrappers so SymPy or exact string matching succeeds."""
    s = expr.strip()
    s = s.strip("$").strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\%", "%")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\dfrac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\b|\D))", "", s)
    s = s.replace("^", "**")
    return s.strip()


def math_answers_equivalent(pred: str, gold: str, tol: float = 1e-6) -> bool:
    """Check whether two math answers are symbolically or numerically equivalent."""
    p_norm = _normalize_latex_math(pred)
    g_norm = _normalize_latex_math(gold)
    if not p_norm:
        return False
    if p_norm.replace(" ", "") == g_norm.replace(" ", ""):
        return True

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transforms = standard_transformations + (implicit_multiplication_application,)
        p_expr = parse_expr(p_norm, transformations=transforms)
        g_expr = parse_expr(g_norm, transformations=transforms)
        diff = sp.simplify(p_expr - g_expr)
        if diff == 0:
            return True
        if diff.is_number:
            return abs(complex(diff.evalf())) < tol
    except Exception:
        pass

    return False


class GroundTruthMathVerifier:
    """Symbolic equivalence verifier for math benchmarks (e.g. MATH-500, GSM8K)."""

    mode_name = "ground_truth_math"
    is_label_free = False

    def verify(self, spec: TaskSpec, solution: TaskSolution) -> VerifyResult:
        if solution.error:
            return VerifyResult(
                task_id=spec.task_id,
                passed=False,
                score=0.0,
                confidence=1.0,
                is_label_free=False,
                failure_type="agent_crash",
                diagnostics=_truncate(solution.error),
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec,
            )

        if not spec.ground_truth or "answer" not in spec.ground_truth:
            raise ValueError(f"Task {spec.task_id} missing ground_truth['answer']")

        gold = str(spec.ground_truth["answer"])
        pred = solution.outputs[0].extracted_answer or extract_boxed_answer(
            solution.outputs[0].raw_text
        )
        ok = math_answers_equivalent(pred, gold)
        return VerifyResult(
            task_id=spec.task_id,
            passed=ok,
            score=1.0 if ok else 0.0,
            confidence=1.0,
            is_label_free=False,
            failure_type="none" if ok else "wrong_answer",
            diagnostics="" if ok else f"Expected {gold!r}, got {pred!r}",
            tokens_used=solution.total_tokens,
            elapsed_sec=solution.elapsed_sec,
        )


register_verifier("ground_truth_code", GroundTruthCodeVerifier)
register_verifier("ground_truth_math", GroundTruthMathVerifier)
