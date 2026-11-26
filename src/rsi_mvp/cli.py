"""CLI entry point for ``rsi``."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rsi_mvp import __version__
from rsi_mvp.agent import solve as default_agent_solve
from rsi_mvp.config import load_dotenv_file, load_experiment_config
from rsi_mvp.doctor import DoctorReport, run_doctor
from rsi_mvp.harness.adapters import get_adapter
from rsi_mvp.harness.archive import ArchiveStore
from rsi_mvp.harness.data import create_dataset_provider
from rsi_mvp.harness.evaluator import SolveFn, evaluate_suite, make_sandboxed_solve_fn
from rsi_mvp.harness.llm import BaseLLMClient, MockLLMClient, OpenAICompatibleClient
from rsi_mvp.harness.loop import EvolutionOrchestrator
from rsi_mvp.harness.sandbox import LLMBackendSpec, create_sandbox
from rsi_mvp.harness.verifiers import get_verifier
from rsi_mvp.schemas import CandidateRecord, EvalSplitType, SplitType


def _format_doctor_report(report: DoctorReport) -> str:
    icons = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}
    lines = ["RSI Agent MVP — Environment Doctor", "=" * 44]
    for check in report.checks:
        lines.append(f"{icons[check.status]} {check.name}: {check.summary}")
        if check.details:
            lines.append(f"       -> {check.details}")
    if report.gpus:
        lines.append("")
        lines.append("GPU Inventory:")
        for gpu in report.gpus:
            tag = "IDLE" if gpu.is_idle else "BUSY"
            lines.append(
                f"  GPU {gpu.index} ({tag}): {gpu.name} | "
                f"{gpu.memory_used_mib}/{gpu.memory_total_mib} MiB | util {gpu.utilization_pct}%"
            )
    return "\n".join(lines)


def _cmd_doctor(args: argparse.Namespace) -> int:
    dotenv = Path(".env")
    if dotenv.is_file():
        load_dotenv_file(dotenv)

    if args.config:
        cfg = load_experiment_config(Path(args.config))
        workspace_root = Path(cfg.experiment.workspace_root)
    else:
        workspace_root = Path(
            os.environ.get("RSI_WORKSPACE_ROOT", str(Path.cwd() / ".workspace"))
        )

    report = run_doctor(
        workspace_root=workspace_root,
        run_gpu_matmul=not args.no_gpu_matmul,
    )
    if args.json:
        print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(_format_doctor_report(report))
    return 1 if report.has_failures else 0


def _cmd_eval(args: argparse.Namespace) -> int:
    dotenv = Path(".env")
    if dotenv.is_file():
        load_dotenv_file(dotenv)

    cfg = load_experiment_config(Path(args.config))
    provider = create_dataset_provider(cfg.dataset)
    adapter = get_adapter(cfg.task.adapter)
    verifier = get_verifier(cfg.task.verifier)

    split: SplitType = args.split
    max_samples = args.max_samples
    eval_split_name: EvalSplitType = (
        "dev_mini"
        if split == "dev" and max_samples is not None
        else ("dev_full" if split == "dev" else split)
    )
    tasks = provider.get_split(
        split=split,
        max_samples=max_samples,
        stratified=True,
        seed=cfg.experiment.seed,
    )

    llm_client: BaseLLMClient
    if args.mock:
        llm_client = MockLLMClient(responses=["```python\npass\n```"])
    else:
        model_name = args.lora or cfg.model.base_model
        base_url = os.environ.get(
            cfg.model.base_url_env,
            f"http://{cfg.model.vllm_host}:{cfg.model.vllm_port}/v1",
        )
        api_key = os.environ.get(cfg.model.api_key_env, "EMPTY")
        llm_client = OpenAICompatibleClient(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            max_total_tokens=cfg.guardrails.max_total_tokens,
        )

    if args.snapshot:
        sandbox = create_sandbox(
            cfg.sandbox, tmp_root=Path(cfg.experiment.workspace_root) / "tmp"
        )
        if args.mock:
            llm_spec = LLMBackendSpec(mode="mock", mock_responses=["```python\npass\n```"])
        else:
            model_name = args.lora or cfg.model.base_model
            base_url = os.environ.get(
                cfg.model.base_url_env,
                f"http://{cfg.model.vllm_host}:{cfg.model.vllm_port}/v1",
            )
            api_key = os.environ.get(cfg.model.api_key_env, "EMPTY")
            llm_spec = LLMBackendSpec(
                mode="openai_http",
                base_url=base_url,
                api_key=api_key,
                model=model_name,
            )
        solve_fn: SolveFn = make_sandboxed_solve_fn(
            snapshot_dir=Path(args.snapshot),
            sandbox=sandbox,
            llm_spec=llm_spec,
            timeout_sec=max(cfg.sandbox.task_timeout_sec, 60),
        )
    else:
        solve_fn = default_agent_solve

    traces_dir = Path(args.output) / "traces" if args.output else None

    summary = evaluate_suite(
        tasks=tasks,
        split_name=eval_split_name,
        solve_fn=solve_fn,
        llm_client=llm_client,
        adapter=adapter,
        verifier=verifier,
        num_samples=cfg.task.num_samples_per_task,
        max_concurrency=args.concurrency or cfg.dataset.max_concurrency,
        traces_dir=traces_dir,
    )

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{eval_split_name}_eval.json").write_text(
            summary.model_dump_json(indent=2),
            encoding="utf-8",
        )

    print(
        f"[EVAL] split={summary.split} verifier={summary.verifier_mode} "
        f"pass={summary.passed_tasks}/{summary.total_tasks} ({summary.pass_rate:.1%}) "
        f"mean_score={summary.mean_score:.4f} tokens={summary.total_tokens} "
        f"failures={summary.failure_breakdown}"
    )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    dotenv = Path(".env")
    if dotenv.is_file():
        load_dotenv_file(dotenv)

    cfg = load_experiment_config(Path(args.config))
    model_ref = args.model or cfg.model.base_model
    gpu_ids = ",".join(str(g) for g in cfg.hardware.serving_gpus)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_ref,
        "--served-model-name",
        cfg.model.base_model,
        "--host",
        cfg.model.vllm_host,
        "--port",
        str(cfg.model.vllm_port),
        "--max-model-len",
        str(cfg.model.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if cfg.model.enable_lora:
        cmd.append("--enable-lora")
    if len(cfg.hardware.serving_gpus) > 1:
        cmd.extend(["--tensor-parallel-size", str(len(cfg.hardware.serving_gpus))])

    env: dict[str, Any] = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    ws = Path(cfg.experiment.workspace_root)
    env.setdefault("HF_HOME", str(ws / "hf_cache"))
    env.setdefault("TMPDIR", str(ws / "tmp"))
    env.setdefault("XDG_CACHE_HOME", str(ws / "tmp" / "xdg_cache"))
    env.setdefault("TRITON_CACHE_DIR", str(ws / "tmp" / "triton_cache"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ws / "tmp" / "inductor_cache"))
    env.setdefault("FLASHINFER_WORKSPACE_DIR", str(ws / "tmp" / "flashinfer_cache"))
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    env.setdefault("VLLM_CACHE_ROOT", str(ws / "tmp" / "vllm_cache"))

    print(f"[SERVE] CUDA_VISIBLE_DEVICES={gpu_ids} {' '.join(cmd)}")
    if args.dry_run:
        return 0
    return subprocess.call(cmd, env=env)


def _fmt_score(val: float | None) -> str:
    return f"{val:.1%}" if val is not None else "-"


def _format_lineage_table(candidates: list[CandidateRecord]) -> str:
    header = (
        f"{'ID':<10} {'Gen':>3} {'Parent':<10} {'Track':<10} {'Status':<22} "
        f"{'Mini':>6} {'Full':>6} {'Anchor':>6} {'Holdout':>7} {'Tokens':>8}  Hypothesis"
    )
    lines = [header, "-" * len(header)]
    for c in candidates:
        hyp = (c.hypothesis[:44] + "...") if len(c.hypothesis) > 47 else c.hypothesis
        lines.append(
            f"{c.candidate_id:<10} {c.generation:>3} {(c.parent_id or '-'):<10} "
            f"{c.track:<10} {c.status:<22} {_fmt_score(c.dev_mini_score):>6} "
            f"{_fmt_score(c.dev_full_score):>6} {_fmt_score(c.anchor_score):>6} "
            f"{_fmt_score(c.holdout_score):>7} {c.total_tokens:>8}  {hyp}"
        )
    return "\n".join(lines)


def _cmd_run(args: argparse.Namespace) -> int:
    dotenv = Path(".env")
    if dotenv.is_file():
        load_dotenv_file(dotenv)

    run_dir: Path | None = Path(args.resume) if args.resume else None
    if run_dir is not None:
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Resume directory missing config.yaml: {cfg_path}")
        cfg = load_experiment_config(cfg_path)
    else:
        if not args.config:
            print("Error: --config is required unless --resume is provided.", file=sys.stderr)
            return 2
        cfg = load_experiment_config(Path(args.config))

    orch = EvolutionOrchestrator(
        config=cfg,
        run_dir=run_dir,
        enforce_storage_guard=not args.skip_storage_guard,
    )
    candidates = orch.run(max_generations=args.max_generations)
    print(f"[RUN] Completed in {orch.run_dir}")
    print(_format_lineage_table(candidates))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    store = ArchiveStore(Path(args.run))
    candidates = store.list_candidates()
    if args.json:
        print(json.dumps([c.model_dump() for c in candidates], indent=2, ensure_ascii=False))
    else:
        print(f"Run directory: {args.run}")
        print(_format_lineage_table(candidates))
        best = store.best_candidate()
        if best is not None:
            print(
                f"\nBest accepted: {best.candidate_id} (gen {best.generation}) | "
                f"dev_full={_fmt_score(best.dev_full_score)} | "
                f"anchor={_fmt_score(best.anchor_score)} | "
                f"holdout={_fmt_score(best.holdout_score)}"
            )
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    store = ArchiveStore(run_dir)
    cand_a = store.get_candidate(args.cand_a)
    cand_b = store.get_candidate(args.cand_b)
    if cand_a is None or cand_b is None:
        missing = args.cand_a if cand_a is None else args.cand_b
        print(f"Candidate not found in {run_dir}: {missing}", file=sys.stderr)
        return 1

    dir_a = run_dir / cand_a.snapshot_dir / "agent"
    dir_b = run_dir / cand_b.snapshot_dir / "agent"
    files_a = (
        {p.relative_to(dir_a).as_posix() for p in dir_a.rglob("*.py")}
        if dir_a.is_dir()
        else set()
    )
    files_b = (
        {p.relative_to(dir_b).as_posix() for p in dir_b.rglob("*.py")}
        if dir_b.is_dir()
        else set()
    )

    print(
        f"Comparing {cand_a.candidate_id} (dev={_fmt_score(cand_a.dev_full_score)}, "
        f"holdout={_fmt_score(cand_a.holdout_score)}) -> "
        f"{cand_b.candidate_id} (dev={_fmt_score(cand_b.dev_full_score)}, "
        f"holdout={_fmt_score(cand_b.holdout_score)})\n"
    )
    diff_chunks: list[str] = []
    for rel in sorted(files_a | files_b):
        text_a = (
            (dir_a / rel).read_text(encoding="utf-8").splitlines(keepends=True)
            if rel in files_a
            else []
        )
        text_b = (
            (dir_b / rel).read_text(encoding="utf-8").splitlines(keepends=True)
            if rel in files_b
            else []
        )
        diff_chunks.extend(
            difflib.unified_diff(
                text_a,
                text_b,
                fromfile=f"{cand_a.candidate_id}/agent/{rel}",
                tofile=f"{cand_b.candidate_id}/agent/{rel}",
            )
        )
    print("".join(diff_chunks) or "(No Python code differences)")
    return 0


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    print(
        f"Command 'rsi {args.command}' is registered in the CLI contract and scheduled for M4–M5.",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsi",
        description="RSI Agent MVP — Dual-track Recursive Self-Improvement CLI",
    )
    parser.add_argument("--version", action="version", version=f"rsi-agent-mvp {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_doctor = subparsers.add_parser("doctor", help="Check local GPU, disk, and Python stack")
    p_doctor.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    p_doctor.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_doctor.add_argument(
        "--no-gpu-matmul",
        action="store_true",
        help="Skip live PyTorch CUDA BF16 matmul smoke check",
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    p_eval = subparsers.add_parser("eval", help="Evaluate agent v0 or a snapshot on a split")
    p_eval.add_argument("--config", type=str, required=True, help="Experiment YAML config path")
    p_eval.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "holdout", "anchor"],
        help="Dataset split to evaluate",
    )
    p_eval.add_argument("--max-samples", type=int, default=None, help="Optional stratified cap")
    p_eval.add_argument("--snapshot", type=str, default=None, help="Candidate snapshot directory")
    p_eval.add_argument("--lora", type=str, default=None, help="Optional served LoRA adapter name")
    p_eval.add_argument("--output", type=str, default=None, help="Directory to write eval & traces")
    p_eval.add_argument("--concurrency", type=int, default=None, help="Override worker concurrency")
    p_eval.add_argument("--mock", action="store_true", help="Use offline MockLLMClient")
    p_eval.set_defaults(func=_cmd_eval)

    p_serve = subparsers.add_parser("serve", help="Start local vLLM OpenAI-compatible server")
    p_serve.add_argument("--config", type=str, required=True, help="Experiment YAML config path")
    p_serve.add_argument("--model", type=str, default=None, help="Override model path/repo")
    p_serve.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    p_serve.add_argument("--dry-run", action="store_true", help="Print command without starting")
    p_serve.set_defaults(func=_cmd_serve)

    p_run = subparsers.add_parser("run", help="Run or resume an RSI evolution experiment")
    p_run.add_argument("--config", type=str, default=None, help="Experiment YAML config path")
    p_run.add_argument("--resume", type=str, default=None, help="Existing run directory to resume")
    p_run.add_argument("--max-generations", type=int, default=None, help="Override max generations")
    p_run.add_argument("--skip-storage-guard", action="store_true", help="Skip disk free guard")
    p_run.set_defaults(func=_cmd_run)

    p_inspect = subparsers.add_parser("inspect", help="Inspect experiment lineage archive")
    p_inspect.add_argument("--run", type=str, required=True, help="Run directory path")
    p_inspect.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_diff = subparsers.add_parser("diff", help="Show code diff between two candidates")
    p_diff.add_argument("--run", type=str, required=True, help="Run directory path")
    p_diff.add_argument("cand_a", type=str, help="First candidate ID (e.g. cand_0000)")
    p_diff.add_argument("cand_b", type=str, help="Second candidate ID (e.g. cand_0002)")
    p_diff.set_defaults(func=_cmd_diff)

    for cmd_name, help_text in [
        ("train-lora", "Run a single-GPU LoRA fine-tuning round from traces (M4)"),
        ("report", "Generate Markdown summary report for a run (M5)"),
    ]:
        p = subparsers.add_parser(cmd_name, help=help_text)
        p.set_defaults(func=_cmd_not_implemented)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
