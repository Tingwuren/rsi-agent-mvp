"""Deterministic ground-truth verifiers for Python code (pytest) and symbolic math."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rsi_mvp.harness.adapters.code import extract_python_code
from rsi_mvp.harness.adapters.math import extract_boxed_answer
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


def _classify_pytest_failure(output: str, timed_out: bool) -> FailureType:
    if timed_out:
        return "timeout"
    if "SyntaxError" in output or "IndentationError" in output:
        return "syntax_error"
    if "AssertionError" in output or "assert " in output:
        return "assertion_error"
    return "runtime_error"


class GroundTruthCodeVerifier:
    """Evaluates generated Python code against immutable pytest tests in a clean temp dir."""

    mode_name = "ground_truth_code"
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

        start = time.perf_counter()
        tmp_base = os.environ.get("TMPDIR")
        with tempfile.TemporaryDirectory(prefix="rsi_eval_", dir=tmp_base) as tmpdir:
            work = Path(tmpdir)
            (work / entry_file).write_text(code, encoding="utf-8")
            (work / "test_solution.py").write_text(test_code, encoding="utf-8")

            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(work),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": tmpdir,
            }
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "--no-header",
                "-o",
                "addopts=",
                str(work / "test_solution.py"),
            ]
            timed_out = False
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=spec.timeout_sec,
                    check=False,
                )
                combined = (proc.stdout + "\n" + proc.stderr).strip()
                returncode = proc.returncode
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                raw_out = exc.stdout
                raw_err = exc.stderr
                out_str = (
                    raw_out.decode("utf-8", "replace")
                    if isinstance(raw_out, bytes)
                    else (raw_out or "")
                )
                err_str = (
                    raw_err.decode("utf-8", "replace")
                    if isinstance(raw_err, bytes)
                    else (raw_err or "")
                )
                combined = f"Timed out after {spec.timeout_sec}s\n{out_str}\n{err_str}".strip()
                returncode = -1

        elapsed = time.perf_counter() - start
        if not timed_out and returncode == 0:
            return VerifyResult(
                task_id=spec.task_id,
                passed=True,
                score=1.0,
                confidence=1.0,
                is_label_free=False,
                failure_type="none",
                diagnostics=_truncate(combined),
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec + elapsed,
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
            failure_type=_classify_pytest_failure(combined, timed_out=timed_out),
            diagnostics=_truncate(combined),
            tokens_used=solution.total_tokens,
            elapsed_sec=solution.elapsed_sec + elapsed,
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
