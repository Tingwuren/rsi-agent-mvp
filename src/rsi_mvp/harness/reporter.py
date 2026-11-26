"""Markdown experiment summary report generator (summary.md)."""

from __future__ import annotations

import difflib
from pathlib import Path

import yaml

from rsi_mvp.harness.archive import ArchiveStore, candidate_primary_score
from rsi_mvp.schemas import CandidateRecord


def _fmt_pct(val: float | None) -> str:
    return f"{val:.1%}" if val is not None else "-"


def _fmt_delta(val: float | None) -> str:
    return f"{val:+.1%}" if val is not None else "-"


def _compute_agent_diff(run_dir: Path, cand_a: CandidateRecord, cand_b: CandidateRecord) -> str:
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
    chunks: list[str] = []
    for rel in sorted(files_a | files_b):
        lines_a = (
            (dir_a / rel).read_text(encoding="utf-8").splitlines(keepends=True)
            if rel in files_a
            else []
        )
        lines_b = (
            (dir_b / rel).read_text(encoding="utf-8").splitlines(keepends=True)
            if rel in files_b
            else []
        )
        chunks.extend(
            difflib.unified_diff(
                lines_a,
                lines_b,
                fromfile=f"{cand_a.candidate_id}/agent/{rel}",
                tofile=f"{cand_b.candidate_id}/agent/{rel}",
            )
        )
    return "".join(chunks)


def generate_summary_report(
    run_dir: Path,
    overfit_warning_threshold: float = 0.15,
    write_file: bool = True,
) -> str:
    """Build a structured Markdown report (``summary.md``) for an RSI experiment run."""
    run_dir = Path(run_dir)
    store = ArchiveStore(run_dir)
    candidates = store.list_candidates()
    loras = store.list_lora_adapters()
    v0 = store.get_candidate("cand_0000")
    best = store.best_candidate()

    cfg_path = run_dir / "config.yaml"
    cfg_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    exp_name = cfg_raw.get("experiment", {}).get("name", run_dir.name)
    base_model = cfg_raw.get("model", {}).get("base_model", "unknown")
    adapter_name = cfg_raw.get("task", {}).get("adapter", "unknown")
    verifier_name = cfg_raw.get("task", {}).get("verifier", "unknown")
    policy_name = cfg_raw.get("evolution", {}).get("selection_policy", "unknown")
    wa_enabled = cfg_raw.get("weight_adaptation", {}).get("enabled", False)

    # Compute high-level scientific metrics
    mutated = [c for c in candidates if c.generation > 0 and c.track == "scaffold"]
    smoke_passed = [
        c for c in mutated if c.status not in ("rejected_invalid", "rejected_smoke_failed", "error")
    ]
    valid_mutation_rate = (len(smoke_passed) / len(mutated)) if mutated else None

    dev_v0 = candidate_primary_score(v0) if v0 else None
    dev_best = candidate_primary_score(best) if best else None
    dev_gain = (dev_best - dev_v0) if (dev_v0 is not None and dev_best is not None) else None

    ho_v0 = v0.holdout_score if v0 else None
    ho_best = best.holdout_score if best else None
    ho_gain = (ho_best - ho_v0) if (ho_v0 is not None and ho_best is not None) else None

    overfit_gap = (dev_gain - ho_gain) if (dev_gain is not None and ho_gain is not None) else None
    anc_v0 = v0.anchor_score if v0 else None
    anc_best = best.anchor_score if best else None
    anc_delta = (anc_best - anc_v0) if (anc_v0 is not None and anc_best is not None) else None

    lines: list[str] = [
        f"# RSI Experiment Report — `{exp_name}`",
        "",
        "## 1. Configuration Overview",
        "",
        f"- **Run Directory**: `{run_dir}`",
        f"- **Base Model**: `{base_model}`",
        f"- **Task / Verifier**: `{adapter_name}` / `{verifier_name}`",
        f"- **Selection Policy**: `{policy_name}`",
        f"- **Track-B LoRA Adaptation**: `{'enabled' if wa_enabled else 'disabled'}`",
        f"- **Total Candidates Recorded**: `{len(candidates)}` (LoRA adapters: `{len(loras)}`)",
        f"- **Cumulative Dev/Anchor Tokens**: `{store.total_tokens_spent():,}`",
        "",
        "## 2. Headline Generalization & Safety Metrics",
        "",
        "| Metric | Baseline (`v0`) | Best (`"
        + (best.candidate_id if best else "-")
        + "`) | Delta |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Dev Score** | {_fmt_pct(dev_v0)} | {_fmt_pct(dev_best)} | {_fmt_delta(dev_gain)} |",
        f"| **Holdout Score (Generalization)** | {_fmt_pct(ho_v0)} | {_fmt_pct(ho_best)} "
        f"| {_fmt_delta(ho_gain)} |",
        f"| **Anchor Score (Anti-Collapse)** | {_fmt_pct(anc_v0)} | {_fmt_pct(anc_best)} "
        f"| {_fmt_delta(anc_delta)} |",
        "",
        f"- **Valid Mutation Rate**: {_fmt_pct(valid_mutation_rate)}",
        f"- **Overfit Gap** (`Dev Gain - Holdout Gain`): {_fmt_delta(overfit_gap)}",
    ]

    if overfit_gap is not None and overfit_gap > overfit_warning_threshold:
        lines.append(
            f"\n> **[WARNING] Overfitting Alert**: Overfit Gap is `{overfit_gap:+.1%}` "
            f"(exceeds `{overfit_warning_threshold:.0%}` threshold). Dev improvements may not "
            "be generalizing to Holdout."
        )

    anchor_rejected = [c for c in candidates if c.status == "rejected_anchor_drop"]
    if anchor_rejected:
        lines.append(
            f"\n> **[NOTE] Anchor Gatekeeper Active**: Blocked `{len(anchor_rejected)}` "
            "candidate(s) due to Anchor regression."
        )

    lines.extend(
        [
            "",
            "## 3. Dual-Track Evolution Lineage",
            "",
            "| ID | Gen | Parent | Track | LoRA | Status | Mini | Full | Anchor | Holdout | "
            "Entropy | Tokens | Hypothesis |",
            "| :--- | :-: | :--- | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :--- |",
        ]
    )
    for c in candidates:
        ent = f"{c.output_entropy:.2f}" if c.output_entropy is not None else "-"
        hyp = c.hypothesis.replace("|", "\\|").replace("\n", " ")
        if len(hyp) > 64:
            hyp = hyp[:61] + "..."
        lines.append(
            f"| `{c.candidate_id}` | {c.generation} | `{c.parent_id or '-'}` | `{c.track}` | "
            f"`{c.lora_adapter_id or '-'}` | `{c.status}` | {_fmt_pct(c.dev_mini_score)} | "
            f"{_fmt_pct(c.dev_full_score)} | {_fmt_pct(c.anchor_score)} | "
            f"{_fmt_pct(c.holdout_score)} | {ent} | {c.total_tokens} | {hyp} |"
        )

    if loras:
        lines.extend(
            [
                "",
                "## 4. Track-B LoRA Adapters",
                "",
                "| Adapter ID | Method | Samples | Train Loss | Anchor Delta | Accepted | Path |",
                "| :--- | :--- | :-: | :-: | :-: | :-: | :--- |",
            ]
        )
        for a in loras:
            loss_s = f"{a.train_loss:.4f}" if a.train_loss is not None else "-"
            lines.append(
                f"| `{a.adapter_id}` | `{a.method}` | {a.train_samples} | {loss_s} | "
                f"{_fmt_delta(a.anchor_delta)} | `{a.accepted}` | `{a.adapter_path}` |"
            )

    if v0 is not None and best is not None and best.candidate_id != v0.candidate_id:
        diff_text = _compute_agent_diff(run_dir, v0, best)
        lines.extend(
            [
                "",
                f"## 5. Best Candidate Code Diff (`{v0.candidate_id}` -> `{best.candidate_id}`)",
                "",
                "```diff",
                diff_text.rstrip() or "(No Python code changes — e.g. weight-only LoRA update)",
                "```",
            ]
        )

    report = "\n".join(lines) + "\n"
    if write_file:
        (run_dir / "summary.md").write_text(report, encoding="utf-8")
    return report
