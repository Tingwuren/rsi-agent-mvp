"""RSI Evolution Orchestrator: resumable main loop with funnel eval & Anchor Gatekeeper."""

from __future__ import annotations

import json
import os
import random
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rsi_mvp.config import check_storage_guard
from rsi_mvp.harness.adapters import TaskAdapter, get_adapter
from rsi_mvp.harness.archive import (
    ArchiveStore,
    SelectionPolicy,
    candidate_primary_score,
    create_selection_policy,
)
from rsi_mvp.harness.data import DatasetProvider, create_dataset_provider
from rsi_mvp.harness.evaluator import (
    SolveFn,
    evaluate_suite,
    evaluate_with_funnel,
    make_sandboxed_solve_fn,
)
from rsi_mvp.harness.llm import BaseLLMClient, BudgetExceededError, MockLLMClient
from rsi_mvp.harness.patcher import apply_and_verify_patch, run_dual_entry_smoke_test
from rsi_mvp.harness.sandbox import LLMBackendSpec, SandboxExecutor, create_sandbox
from rsi_mvp.harness.verifiers import GroundTruthCodeVerifier, Verifier, get_verifier
from rsi_mvp.schemas import (
    CandidateRecord,
    CandidateStatus,
    ExperimentConfig,
    MutationProposal,
    SuiteEvalSummary,
    TaskSpec,
)

_DEFAULT_SEED_AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"

SolveFactory = Callable[[int, Path], tuple[SolveFn, BaseLLMClient]]
ImproveSpecFactory = Callable[[int, CandidateRecord], LLMBackendSpec]


def _read_agent_sources(agent_dir: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in sorted(agent_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(agent_dir).as_posix()
        sources[rel] = path.read_text(encoding="utf-8")
    return sources


def _sample_failed_cases(
    parent_snapshot_dir: Path, max_cases: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load parent dev summary and sample up to ``max_cases`` failed dev traces only."""
    eval_file = parent_snapshot_dir / "dev_full_eval.json"
    if not eval_file.is_file():
        eval_file = parent_snapshot_dir / "dev_mini_eval.json"
    if not eval_file.is_file():
        return {}, []

    summary = SuiteEvalSummary.model_validate_json(eval_file.read_text(encoding="utf-8"))
    summary_dict = {
        "split": summary.split,
        "pass_rate": summary.pass_rate,
        "mean_score": summary.mean_score,
        "passed_tasks": summary.passed_tasks,
        "total_tasks": summary.total_tasks,
        "failure_breakdown": summary.failure_breakdown,
    }

    failed_results = [r for r in summary.results if not r.passed][:max_cases]
    failed_cases: list[dict[str, Any]] = []
    traces_dir = parent_snapshot_dir / "traces"
    for r in failed_results:
        case: dict[str, Any] = {
            "task_id": r.task_id,
            "failure_type": r.failure_type,
            "score": r.score,
            "diagnostics": r.diagnostics[:1200],
        }
        trace_file = traces_dir / f"{r.task_id}.json"
        if trace_file.is_file():
            tr = json.loads(trace_file.read_text(encoding="utf-8"))
            outputs = tr.get("solution", {}).get("outputs") or []
            if outputs:
                case["extracted_answer"] = str(outputs[0].get("extracted_answer", ""))[:1000]
        failed_cases.append(case)

    return summary_dict, failed_cases


def _safe_load_split(
    provider: DatasetProvider,
    split: Any,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[TaskSpec]:
    try:
        return provider.get_split(split=split, max_samples=max_samples, stratified=True, seed=seed)
    except FileNotFoundError:
        return []


class EvolutionOrchestrator:
    """Drives the resumable Scaffold RSI evolution loop with funnel eval and Anchor Gatekeeper."""

    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: Path | None = None,
        seed_agent_dir: Path | None = None,
        sandbox: SandboxExecutor | None = None,
        provider: DatasetProvider | None = None,
        adapter: TaskAdapter | None = None,
        verifier: Verifier | None = None,
        policy: SelectionPolicy | None = None,
        solve_factory: SolveFactory | None = None,
        improve_spec_factory: ImproveSpecFactory | None = None,
        enforce_storage_guard: bool = True,
    ) -> None:
        self.config = config
        self.seed_agent_dir = Path(seed_agent_dir or _DEFAULT_SEED_AGENT_DIR)

        if run_dir is None:
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(config.experiment.output_dir) / f"{ts}_{config.experiment.name}"
        else:
            self.run_dir = Path(run_dir)

        if enforce_storage_guard:
            check_storage_guard(
                self.run_dir, min_free_gb=config.guardrails.min_free_disk_gb
            )

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.run_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Freeze config on initial creation
        cfg_file = self.run_dir / "config.yaml"
        if not cfg_file.is_file():
            cfg_file.write_text(
                yaml.safe_dump(config.model_dump(), sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

        self.archive = ArchiveStore(self.run_dir)
        tmp_root = Path(config.experiment.workspace_root) / "tmp"
        self.sandbox: SandboxExecutor = sandbox or create_sandbox(
            config.sandbox, tmp_root=tmp_root
        )
        self.provider: DatasetProvider = provider or create_dataset_provider(config.dataset)
        self.adapter: TaskAdapter = adapter or get_adapter(config.task.adapter)
        if verifier is not None:
            self.verifier = verifier
        elif config.task.verifier == "ground_truth_code":
            self.verifier = GroundTruthCodeVerifier(sandbox=self.sandbox)
        else:
            self.verifier = get_verifier(config.task.verifier)

        self.policy: SelectionPolicy = policy or create_selection_policy(
            config.evolution.selection_policy
        )
        self._solve_factory = solve_factory
        self._improve_spec_factory = improve_spec_factory
        self._rng = random.Random(config.experiment.seed)

        # Pre-load deterministic splits
        seed = config.experiment.seed
        self.dev_full_tasks = self.provider.get_split("dev", stratified=True, seed=seed)
        self.dev_mini_tasks = self.provider.get_split(
            "dev",
            max_samples=min(config.dataset.mini_eval_size, len(self.dev_full_tasks))
            if self.dev_full_tasks
            else None,
            stratified=True,
            seed=seed,
        )
        self.holdout_tasks = _safe_load_split(self.provider, "holdout", seed=seed)
        self.anchor_tasks = (
            _safe_load_split(self.provider, "anchor", seed=seed)
            if config.guardrails.anchor_enabled
            else []
        )

    def _default_llm_spec(self, active_lora: str | None = None) -> LLMBackendSpec:
        base_url = os.environ.get(
            self.config.model.base_url_env,
            f"http://{self.config.model.vllm_host}:{self.config.model.vllm_port}/v1",
        )
        api_key = os.environ.get(self.config.model.api_key_env, "EMPTY")
        return LLMBackendSpec(
            mode="openai_http",
            base_url=base_url,
            api_key=api_key,
            model=active_lora or self.config.model.base_model,
        )

    def _build_solve_runner(
        self, generation: int, snapshot_dir: Path, active_lora: str | None = None
    ) -> tuple[SolveFn, BaseLLMClient]:
        if self._solve_factory is not None:
            return self._solve_factory(generation, snapshot_dir)
        llm_spec = self._default_llm_spec(active_lora=active_lora)
        solve_fn = make_sandboxed_solve_fn(
            snapshot_dir=snapshot_dir,
            sandbox=self.sandbox,
            llm_spec=llm_spec,
            timeout_sec=max(self.config.sandbox.task_timeout_sec, 60),
        )
        return solve_fn, MockLLMClient()

    def _build_improve_spec(
        self, generation: int, parent: CandidateRecord
    ) -> LLMBackendSpec:
        if self._improve_spec_factory is not None:
            return self._improve_spec_factory(generation, parent)
        return self._default_llm_spec(active_lora=parent.lora_adapter_id)

    def _baseline_anchor_score(self) -> float | None:
        v0 = self.archive.get_candidate("cand_0000")
        return v0.anchor_score if v0 is not None else None

    def _bootstrap_v0(self) -> CandidateRecord:
        existing = self.archive.get_candidate("cand_0000")
        if existing is not None:
            return existing

        snap_dir = self.snapshots_dir / "cand_0000"
        if (snap_dir / "agent").exists():
            shutil.rmtree(snap_dir / "agent")
        snap_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.seed_agent_dir, snap_dir / "agent")

        smoke_ok, smoke_err = run_dual_entry_smoke_test(
            candidate_snapshot_dir=snap_dir,
            sandbox=self.sandbox,
            timeout_sec=20,
        )
        if not smoke_ok:
            rec = CandidateRecord(
                candidate_id="cand_0000",
                generation=0,
                parent_id=None,
                track="bootstrap",
                status="rejected_smoke_failed",
                snapshot_dir="snapshots/cand_0000",
                hypothesis="Initial v0 baseline",
                failure_reason=smoke_err,
            )
            self.archive.add_candidate(rec)
            raise RuntimeError(f"Bootstrap v0 failed dual-entry smoke test: {smoke_err}")

        solve_fn, llm_client = self._build_solve_runner(generation=0, snapshot_dir=snap_dir)
        dev_summary = evaluate_suite(
            tasks=self.dev_full_tasks,
            split_name="dev_full",
            solve_fn=solve_fn,
            llm_client=llm_client,
            adapter=self.adapter,
            verifier=self.verifier,
            num_samples=self.config.task.num_samples_per_task,
            max_concurrency=self.config.dataset.max_concurrency,
            traces_dir=snap_dir / "traces",
        )
        (snap_dir / "dev_full_eval.json").write_text(
            dev_summary.model_dump_json(indent=2), encoding="utf-8"
        )
        total_tokens = dev_summary.total_tokens

        anchor_score: float | None = None
        if self.config.guardrails.anchor_enabled and self.anchor_tasks:
            anchor_summary = evaluate_suite(
                tasks=self.anchor_tasks,
                split_name="anchor",
                solve_fn=solve_fn,
                llm_client=llm_client,
                adapter=self.adapter,
                verifier=self.verifier,
                num_samples=self.config.task.num_samples_per_task,
                max_concurrency=self.config.dataset.max_concurrency,
                traces_dir=None,
            )
            (snap_dir / "anchor_eval.json").write_text(
                anchor_summary.model_dump_json(indent=2), encoding="utf-8"
            )
            anchor_score = anchor_summary.pass_rate
            total_tokens += anchor_summary.total_tokens

        holdout_score: float | None = None
        if self.holdout_tasks:
            holdout_summary = evaluate_suite(
                tasks=self.holdout_tasks,
                split_name="holdout",
                solve_fn=solve_fn,
                llm_client=llm_client,
                adapter=self.adapter,
                verifier=self.verifier,
                num_samples=self.config.task.num_samples_per_task,
                max_concurrency=self.config.dataset.max_concurrency,
                traces_dir=None,
            )
            (snap_dir / "holdout_eval.json").write_text(
                holdout_summary.model_dump_json(indent=2), encoding="utf-8"
            )
            holdout_score = holdout_summary.pass_rate

        rec = CandidateRecord(
            candidate_id="cand_0000",
            generation=0,
            parent_id=None,
            track="bootstrap",
            status="accepted",
            snapshot_dir="snapshots/cand_0000",
            hypothesis="Initial v0 baseline",
            dev_mini_score=dev_summary.pass_rate,
            dev_full_score=dev_summary.pass_rate,
            anchor_score=anchor_score,
            holdout_score=holdout_score,
            total_tokens=total_tokens,
        )
        self.archive.add_candidate(rec)
        self.archive.record_event(
            "bootstrap_complete",
            {
                "candidate_id": rec.candidate_id,
                "dev_full_score": rec.dev_full_score,
                "anchor_score": rec.anchor_score,
                "holdout_score": rec.holdout_score,
            },
        )
        return rec

    def _run_one_generation(self, generation: int) -> CandidateRecord:
        accepted_pool = self.archive.accepted_candidates()
        parent = self.policy.select_parent(accepted_pool, self._rng)
        self.archive.increment_selection_count(parent.candidate_id)

        cand_id = self.archive.next_candidate_id()
        parent_snap = self.run_dir / parent.snapshot_dir
        cand_snap = self.snapshots_dir / cand_id
        cand_snap.mkdir(parents=True, exist_ok=True)
        rel_snap = f"snapshots/{cand_id}"

        # Build meta-agent context strictly from parent's Dev traces + source files
        dev_summary, failed_cases = _sample_failed_cases(
            parent_snap, max_cases=self.config.evolution.max_failed_traces_in_context
        )
        recent_history = [
            {
                "candidate_id": c.candidate_id,
                "generation": c.generation,
                "status": c.status,
                "hypothesis": c.hypothesis,
                "dev_score": candidate_primary_score(c),
                "failure_reason": c.failure_reason,
            }
            for c in self.archive.list_candidates()[-5:]
        ]
        improve_context = {
            "parent_id": parent.candidate_id,
            "dev_summary": dev_summary,
            "failed_cases": failed_cases,
            "source_files": _read_agent_sources(parent_snap / "agent"),
            "recent_history": recent_history,
        }

        improve_spec = self._build_improve_spec(generation, parent)
        imp_res = self.sandbox.run_agent_improve(
            snapshot_dir=parent_snap,
            improve_context=improve_context,
            llm_spec=improve_spec,
            timeout_sec=self.config.sandbox.improve_timeout_sec,
        )
        if not imp_res.ok or imp_res.payload is None:
            rec = CandidateRecord(
                candidate_id=cand_id,
                generation=generation,
                parent_id=parent.candidate_id,
                track="scaffold",
                lora_adapter_id=parent.lora_adapter_id,
                status="error",
                snapshot_dir=rel_snap,
                failure_reason=f"Improver failed: {imp_res.stderr or imp_res.stdout}",
            )
            self.archive.add_candidate(rec)
            return rec

        try:
            proposal = MutationProposal.model_validate(imp_res.payload)
        except Exception as exc:
            rec = CandidateRecord(
                candidate_id=cand_id,
                generation=generation,
                parent_id=parent.candidate_id,
                track="scaffold",
                lora_adapter_id=parent.lora_adapter_id,
                status="rejected_invalid",
                snapshot_dir=rel_snap,
                failure_reason=f"Invalid MutationProposal schema: {exc}",
            )
            self.archive.add_candidate(rec)
            return rec

        (cand_snap / "proposal.json").write_text(
            proposal.model_dump_json(indent=2), encoding="utf-8"
        )

        # Apply patch + AST/boundary check + dual-entry smoke test
        patch_res = apply_and_verify_patch(
            parent_snapshot_dir=parent_snap,
            candidate_snapshot_dir=cand_snap,
            proposal=proposal,
            sandbox=self.sandbox,
            max_files_per_mutation=self.config.evolution.max_files_per_mutation,
            smoke_timeout_sec=min(20, self.config.sandbox.task_timeout_sec),
        )
        if not patch_res.ok:
            fail_status: CandidateStatus = (
                "rejected_smoke_failed"
                if patch_res.failure_stage == "smoke"
                else "rejected_invalid"
            )
            rec = CandidateRecord(
                candidate_id=cand_id,
                generation=generation,
                parent_id=parent.candidate_id,
                track="scaffold",
                lora_adapter_id=parent.lora_adapter_id,
                status=fail_status,
                snapshot_dir=rel_snap,
                diff_path=f"{rel_snap}/patch.diff" if patch_res.diff_path else None,
                hypothesis=proposal.hypothesis,
                failure_reason=f"[{patch_res.failure_stage}] {patch_res.error}",
            )
            self.archive.add_candidate(rec)
            return rec

        # Two-stage Mini -> Full Dev funnel + Anchor Gatekeeper
        solve_fn, llm_client = self._build_solve_runner(
            generation=generation,
            snapshot_dir=cand_snap,
            active_lora=parent.lora_adapter_id,
        )
        existing_mini_scores = [
            c.dev_mini_score
            for c in accepted_pool
            if c.dev_mini_score is not None
        ]
        funnel = evaluate_with_funnel(
            dev_mini_tasks=self.dev_mini_tasks,
            dev_full_tasks=self.dev_full_tasks,
            anchor_tasks=self.anchor_tasks,
            solve_fn=solve_fn,
            llm_client=llm_client,
            adapter=self.adapter,
            verifier=self.verifier,
            snapshot_dir=cand_snap,
            existing_mini_scores=existing_mini_scores,
            full_eval_top_k=self.config.dataset.full_eval_top_k,
            anchor_enabled=self.config.guardrails.anchor_enabled,
            baseline_anchor_score=self._baseline_anchor_score(),
            max_anchor_drop=self.config.guardrails.max_anchor_drop,
            num_samples=self.config.task.num_samples_per_task,
            max_concurrency=self.config.dataset.max_concurrency,
        )

        best_before = self.archive.best_candidate()
        best_score_before = candidate_primary_score(best_before) if best_before else -1.0
        dev_full_score = funnel.dev_full.pass_rate if funnel.dev_full else None
        anchor_score = funnel.anchor.pass_rate if funnel.anchor else None
        primary_score = (
            dev_full_score if dev_full_score is not None else funnel.dev_mini.pass_rate
        )
        total_tokens = funnel.total_tokens

        status: CandidateStatus
        failure_reason: str | None = None
        if not funnel.promoted_to_full:
            status = "rejected_mini_pruned"
            failure_reason = (
                f"dev_mini_score={funnel.dev_mini.pass_rate:.4f} did not make top-"
                f"{self.config.dataset.full_eval_top_k}"
            )
        elif not funnel.anchor_passed:
            status = "rejected_anchor_drop"
            failure_reason = (
                f"Anchor score dropped by {funnel.anchor_delta:+.4f} "
                f"(exceeds max_anchor_drop={self.config.guardrails.max_anchor_drop:.4f})"
            )
        elif self.policy.should_accept(primary_score, total_tokens, best_before):
            status = "accepted"
        else:
            status = "rejected_low_score"
            failure_reason = (
                f"score={primary_score:.4f} did not beat best={best_score_before:.4f}"
            )

        # Periodic or new-best Holdout evaluation (black-box, never fed back to improver)
        holdout_score: float | None = None
        is_new_best = status == "accepted" and primary_score > best_score_before + 1e-9
        is_periodic = (
            generation % self.config.evolution.holdout_eval_interval == 0
            or generation == self.config.evolution.max_generations
        )
        if self.holdout_tasks and status == "accepted" and (is_new_best or is_periodic):
            ho_summary = evaluate_suite(
                tasks=self.holdout_tasks,
                split_name="holdout",
                solve_fn=solve_fn,
                llm_client=llm_client,
                adapter=self.adapter,
                verifier=self.verifier,
                num_samples=self.config.task.num_samples_per_task,
                max_concurrency=self.config.dataset.max_concurrency,
                traces_dir=None,
            )
            (cand_snap / "holdout_eval.json").write_text(
                ho_summary.model_dump_json(indent=2), encoding="utf-8"
            )
            holdout_score = ho_summary.pass_rate

        rec = CandidateRecord(
            candidate_id=cand_id,
            generation=generation,
            parent_id=parent.candidate_id,
            track="scaffold",
            lora_adapter_id=parent.lora_adapter_id,
            status=status,
            snapshot_dir=rel_snap,
            diff_path=f"{rel_snap}/patch.diff" if patch_res.diff_path else None,
            hypothesis=proposal.hypothesis,
            dev_mini_score=funnel.dev_mini.pass_rate,
            dev_full_score=dev_full_score,
            anchor_score=anchor_score,
            holdout_score=holdout_score,
            total_tokens=total_tokens,
            failure_reason=failure_reason,
        )
        self.archive.add_candidate(rec)
        return rec

    def run(self, max_generations: int | None = None) -> list[CandidateRecord]:
        """Run (or resume) the evolution loop up to ``max_generations``."""
        target_gens = (
            max_generations
            if max_generations is not None
            else self.config.evolution.max_generations
        )
        self._bootstrap_v0()

        start_gen = max(1, self.archive.max_generation() + 1)
        no_improve_streak = 0
        best_seen = self.archive.best_candidate()
        best_score = candidate_primary_score(best_seen) if best_seen else 0.0

        for gen in range(start_gen, target_gens + 1):
            if self.archive.total_tokens_spent() >= self.config.guardrails.max_total_tokens:
                self.archive.record_event(
                    "stopped_budget",
                    {"generation": gen, "tokens": self.archive.total_tokens_spent()},
                )
                break
            if no_improve_streak >= self.config.evolution.early_stop_patience:
                self.archive.record_event(
                    "stopped_early",
                    {"generation": gen, "patience": no_improve_streak},
                )
                break

            try:
                rec = self._run_one_generation(gen)
            except BudgetExceededError as exc:
                self.archive.record_event("budget_exceeded", {"generation": gen, "error": str(exc)})
                break
            except Exception as exc:
                cand_id = self.archive.next_candidate_id()
                rec = CandidateRecord(
                    candidate_id=cand_id,
                    generation=gen,
                    parent_id=None,
                    track="scaffold",
                    status="error",
                    snapshot_dir=f"snapshots/{cand_id}",
                    failure_reason=f"Unhandled generation error: {type(exc).__name__}: {exc}",
                )
                self.archive.add_candidate(rec)

            self.archive.record_event(
                "generation_complete",
                {
                    "generation": gen,
                    "candidate_id": rec.candidate_id,
                    "parent_id": rec.parent_id,
                    "status": rec.status,
                    "dev_mini_score": rec.dev_mini_score,
                    "dev_full_score": rec.dev_full_score,
                    "anchor_score": rec.anchor_score,
                    "holdout_score": rec.holdout_score,
                },
            )

            if rec.status == "accepted" and candidate_primary_score(rec) > best_score + 1e-9:
                best_score = candidate_primary_score(rec)
                no_improve_streak = 0
            else:
                no_improve_streak += 1

        return self.archive.list_candidates()
