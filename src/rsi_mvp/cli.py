"""CLI entry point for ``rsi``."""

from __future__ import annotations

import argparse
import importlib.util
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
from rsi_mvp.harness.data import create_dataset_provider
from rsi_mvp.harness.evaluator import SolveFn, evaluate_suite
from rsi_mvp.harness.llm import BaseLLMClient, MockLLMClient, OpenAICompatibleClient
from rsi_mvp.harness.verifiers import get_verifier
from rsi_mvp.schemas import EvalSplitType, SplitType


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


def _load_snapshot_solve(snapshot_dir: Path) -> SolveFn:
    """Dynamically load ``solve`` from a candidate snapshot's ``agent/core.py``."""
    agent_init = snapshot_dir / "agent" / "__init__.py"
    if not agent_init.is_file():
        raise FileNotFoundError(f"Snapshot agent package not found: {agent_init}")
    spec = importlib.util.spec_from_file_location("rsi_snapshot_agent", agent_init)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load snapshot from {agent_init}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve


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
        "dev_mini" if split == "dev" and max_samples is not None else (
            "dev_full" if split == "dev" else split
        )
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

    solve_fn: SolveFn = (
        _load_snapshot_solve(Path(args.snapshot)) if args.snapshot else default_agent_solve
    )
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
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    env.setdefault("VLLM_CACHE_ROOT", str(ws / "tmp" / "vllm_cache"))
    env.setdefault("XDG_CACHE_HOME", str(ws / "tmp" / "xdg_cache"))
    env.setdefault("TRITON_CACHE_DIR", str(ws / "tmp" / "triton"))
    env.setdefault("VLLM_CACHE_ROOT", str(ws / "tmp" / "vllm_cache"))
    env.setdefault("FLASHINFER_WORKSPACE_BASE", str(ws / "tmp" / "flashinfer"))

    print(f"[SERVE] CUDA_VISIBLE_DEVICES={gpu_ids} {' '.join(cmd)}")
    if args.dry_run:
        return 0
    return subprocess.call(cmd, env=env)


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    print(
        f"Command 'rsi {args.command}' is registered in the CLI contract and scheduled for M3–M5.",
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

    for cmd_name, help_text in [
        ("run", "Run or resume an RSI evolution experiment (M3)"),
        ("train-lora", "Run a single-GPU LoRA fine-tuning round from traces (M4)"),
        ("inspect", "Inspect experiment lineage archive and metrics (M3/M5)"),
        ("diff", "Show unified code diff between two candidates (M5)"),
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
