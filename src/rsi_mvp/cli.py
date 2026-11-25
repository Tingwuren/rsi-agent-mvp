"""CLI entry point for ``rsi``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rsi_mvp import __version__
from rsi_mvp.config import load_dotenv_file, load_experiment_config
from rsi_mvp.doctor import DoctorReport, run_doctor


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


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    print(
        f"Command 'rsi {args.command}' is registered in the CLI contract and scheduled for M1–M5.",
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

    for cmd_name, help_text in [
        ("serve", "Manage local vLLM inference server (M1/M4)"),
        ("run", "Run or resume an RSI evolution experiment (M3)"),
        ("eval", "Evaluate a (snapshot, lora) candidate on a split (M1/M3)"),
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
