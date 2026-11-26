"""Concurrent evaluation engine coordinating TaskAdapter, Agent, and Verifier."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rsi_mvp.harness.adapters import TaskAdapter
from rsi_mvp.harness.llm import BaseLLMClient, BudgetExceededError
from rsi_mvp.harness.verifiers import Verifier
from rsi_mvp.schemas import (
    CandidateOutput,
    EvalSplitType,
    SuiteEvalSummary,
    TaskSolution,
    TaskSpec,
    VerifyResult,
)

SolveFn = Callable[[dict[str, Any], BaseLLMClient], dict[str, Any]]


def evaluate_single_task(
    spec: TaskSpec,
    solve_fn: SolveFn,
    llm_client: BaseLLMClient,
    adapter: TaskAdapter,
    verifier: Verifier,
    num_samples: int = 1,
) -> tuple[TaskSolution, VerifyResult]:
    """Run ``solve_fn`` on a sanitized ``TaskInput`` and verify the resulting solution."""
    task_input = adapter.to_task_input(spec, num_samples=num_samples)
    start = time.perf_counter()
    try:
        raw_sol = solve_fn(task_input.model_dump(), llm_client)
        solution = TaskSolution.model_validate(raw_sol)
        # Ensure extracted_answer is populated using the domain adapter if agent left it empty
        enriched_outputs: list[CandidateOutput] = []
        for out in solution.outputs:
            extracted = out.extracted_answer or adapter.extract_answer(out.raw_text, spec)
            enriched_outputs.append(
                CandidateOutput(
                    raw_text=out.raw_text,
                    extracted_answer=extracted,
                    auxiliary=out.auxiliary,
                )
            )
        solution = solution.model_copy(update={"outputs": enriched_outputs})
    except BudgetExceededError:
        raise
    except Exception as exc:
        elapsed = time.perf_counter() - start
        solution = TaskSolution(
            task_id=spec.task_id,
            outputs=[CandidateOutput(raw_text="", extracted_answer="")],
            error=f"{type(exc).__name__}: {exc}",
            elapsed_sec=elapsed,
        )

    verify_result = verifier.verify(spec, solution)
    return solution, verify_result


def evaluate_suite(
    tasks: list[TaskSpec],
    split_name: EvalSplitType,
    solve_fn: SolveFn,
    llm_client: BaseLLMClient,
    adapter: TaskAdapter,
    verifier: Verifier,
    num_samples: int = 1,
    max_concurrency: int = 4,
    traces_dir: Path | None = None,
) -> SuiteEvalSummary:
    """Evaluate a list of tasks concurrently and optionally persist traces."""
    if traces_dir is not None:
        traces_dir.mkdir(parents=True, exist_ok=True)

    if not tasks:
        return SuiteEvalSummary(
            split=split_name,
            verifier_mode=verifier.mode_name,
            mean_score=0.0,
            pass_rate=0.0,
            avg_confidence=1.0,
            total_tasks=0,
            passed_tasks=0,
            total_tokens=0,
            avg_elapsed_sec=0.0,
            failure_breakdown={},
            results=[],
        )

    ordered_results: dict[str, tuple[TaskSolution, VerifyResult]] = {}

    if max_concurrency <= 1 or len(tasks) == 1:
        for spec in tasks:
            ordered_results[spec.task_id] = evaluate_single_task(
                spec=spec,
                solve_fn=solve_fn,
                llm_client=llm_client,
                adapter=adapter,
                verifier=verifier,
                num_samples=num_samples,
            )
    else:
        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(tasks))) as pool:
            future_map = {
                pool.submit(
                    evaluate_single_task,
                    spec,
                    solve_fn,
                    llm_client,
                    adapter,
                    verifier,
                    num_samples,
                ): spec.task_id
                for spec in tasks
            }
            for fut in as_completed(future_map):
                task_id = future_map[fut]
                ordered_results[task_id] = fut.result()

    results: list[VerifyResult] = []
    for spec in tasks:
        sol, res = ordered_results[spec.task_id]
        results.append(res)
        if traces_dir is not None:
            trace_payload = {
                "task_id": spec.task_id,
                "domain": spec.domain,
                "category": spec.category,
                "solution": sol.model_dump(),
                "verify": res.model_dump(),
            }
            (traces_dir / f"{spec.task_id}.json").write_text(
                json.dumps(trace_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    total_tasks = len(results)
    passed_tasks = sum(1 for r in results if r.passed)
    mean_score = sum(r.score for r in results) / total_tasks
    pass_rate = passed_tasks / total_tasks
    avg_confidence = sum(r.confidence for r in results) / total_tasks
    total_tokens = sum(r.tokens_used for r in results)
    avg_elapsed = sum(r.elapsed_sec for r in results) / total_tasks
    failures: Counter[str] = Counter(r.failure_type for r in results if r.failure_type != "none")

    return SuiteEvalSummary(
        split=split_name,
        verifier_mode=verifier.mode_name,
        mean_score=round(mean_score, 4),
        pass_rate=round(pass_rate, 4),
        avg_confidence=round(avg_confidence, 4),
        total_tasks=total_tasks,
        passed_tasks=passed_tasks,
        total_tokens=total_tokens,
        avg_elapsed_sec=round(avg_elapsed, 4),
        failure_breakdown=dict(failures),
        results=results,
    )
