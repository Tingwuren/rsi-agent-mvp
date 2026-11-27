#!/usr/bin/env python3
"""Run the upgraded 6-group controlled RSI ablation suite v2 (Exp-1 ~ Exp-6) on GPU 1.

Upgrades over v1:
  1. Rich truncation & token telemetry in failed_cases so Meta-Agent sees real bottlenecks.
  2. Escape-free XML <edit> blocks + 1-shot error-feedback retry per generation.
  3. Concurrent Train collection + un-truncated concise SFT (max_length=4096, lr=2e-5,
     1 epoch, lora_r=8, full Train=100 split for ~65+ verified samples).
  4. Deeper evolution search (4 gens for Exp-2/3/5, 3 gens for Exp-6) on larger Holdout=30.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from rsi_mvp.config import load_dotenv_file
from rsi_mvp.harness.archive import candidate_primary_score
from rsi_mvp.harness.data import JSONLDatasetProvider
from rsi_mvp.harness.lifecycle import VLLMLifecycleManager
from rsi_mvp.harness.loop import EvolutionOrchestrator
from rsi_mvp.harness.reporter import generate_summary_report
from rsi_mvp.harness.trainer import TRLLoRASFTTrainer
from rsi_mvp.schemas import (
    DatasetConfig,
    EvolutionConfig,
    ExperimentConfig,
    ExperimentMetaConfig,
    GuardrailsConfig,
    HardwareConfig,
    ModelConfig,
    SandboxConfig,
    TaskConfig,
    WeightAdaptationConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT / ".workspace"
MODEL_DIR = WORKSPACE_ROOT / "models" / "Qwen3-8B"
SRC_MATH500 = WORKSPACE_ROOT / "datasets" / "math_500"
LABELED_DS = WORKSPACE_ROOT / "datasets" / "ablation_math500_v2_labeled"
UNLABELED_DS = WORKSPACE_ROOT / "datasets" / "ablation_math500_v2_unlabeled"
SUITE_ROOT = WORKSPACE_ROOT / "runs" / "ablation_6exp_math500_v2"


def prepare_ablation_splits(seed: int = 42) -> None:
    """Create matched labeled (Train=100) & unlabeled (Train=60) stratified MATH-500 splits."""
    provider = JSONLDatasetProvider(SRC_MATH500)
    LABELED_DS.mkdir(parents=True, exist_ok=True)
    UNLABELED_DS.mkdir(parents=True, exist_ok=True)

    labeled_sizes = {"train": 100, "dev": 20, "anchor": 16, "holdout": 30}
    unlabeled_sizes = {"train": 60, "dev": 20, "anchor": 16, "holdout": 30}
    for split, n in labeled_sizes.items():
        items = provider.get_split(split, max_samples=n, stratified=True, seed=seed)  # type: ignore[arg-type]
        (LABELED_DS / f"{split}.jsonl").write_text(
            "\n".join(t.model_dump_json() for t in items) + "\n", encoding="utf-8"
        )
        u_items = items[: unlabeled_sizes[split]]
        if split in ("train", "dev"):
            u_items = [t.model_copy(update={"ground_truth": None}) for t in u_items]
        (UNLABELED_DS / f"{split}.jsonl").write_text(
            "\n".join(t.model_dump_json() for t in u_items) + "\n", encoding="utf-8"
        )
    print(
        f"[ablation-v2] Prepared splits: labeled={labeled_sizes}, unlabeled={unlabeled_sizes}",
        flush=True,
    )


def make_base_config(
    exp_name: str,
    dataset_dir: Path,
    *,
    verifier: str = "ground_truth_math",
    num_samples: int = 1,
    max_generations: int = 4,
    allowed_edit_files: list[str] | None = None,
    weight_enabled: bool = False,
    trigger_every_n: int = 4,
    min_train_samples: int = 12,
    min_confidence: float = 0.75,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentMetaConfig(
            name=exp_name,
            seed=42,
            workspace_root=str(WORKSPACE_ROOT),
            output_dir=str(SUITE_ROOT),
        ),
        hardware=HardwareConfig(
            mode="single_gpu_timeshare",
            serving_gpus=[1],
            training_gpus=[1],
        ),
        model=ModelConfig(
            base_model=str(MODEL_DIR),
            vllm_host="127.0.0.1",
            vllm_port=8000,
            max_model_len=16384,
            enable_lora=True,
            temperature=0.0 if num_samples == 1 else 0.7,
        ),
        task=TaskConfig(
            adapter="math",
            verifier=verifier,  # type: ignore[arg-type]
            num_samples_per_task=num_samples,
        ),
        dataset=DatasetConfig(
            provider="jsonl",
            path=str(dataset_dir),
            mini_eval_size=20,
            full_eval_top_k=2,
            max_concurrency=8,
        ),
        evolution=EvolutionConfig(
            max_generations=max_generations,
            selection_policy="evolutionary_archive",
            holdout_eval_interval=2,
            early_stop_patience=5,
            max_files_per_mutation=2,
            max_failed_traces_in_context=4,
            allowed_edit_files=allowed_edit_files,
        ),
        weight_adaptation=WeightAdaptationConfig(
            enabled=weight_enabled,
            trigger_every_n_generations=trigger_every_n,
            method="sft_lora",
            min_train_samples=min_train_samples,
            min_confidence=min_confidence,
            lora_r=8,
            lora_alpha=16,
            learning_rate=2.0e-5,
            num_epochs=1,
            replay_ratio=0.0,
        ),
        guardrails=GuardrailsConfig(
            anchor_enabled=True,
            max_anchor_drop=0.08,
            max_total_tokens=10_000_000,
            min_free_disk_gb=5.0,
        ),
        sandbox=SandboxConfig(
            backend="subprocess",
            task_timeout_sec=150,
            improve_timeout_sec=180,
        ),
    )


def seed_v0_from_exp1(exp1_dir: Path, target_run_dir: Path) -> None:
    if target_run_dir.exists():
        shutil.rmtree(target_run_dir)
    target_run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        exp1_dir / "snapshots" / "cand_0000", target_run_dir / "snapshots" / "cand_0000"
    )
    shutil.copy2(exp1_dir / "archive.db", target_run_dir / "archive.db")


def summarize_experiment(
    exp_id: str,
    label: str,
    run_dir: Path,
    orch: EvolutionOrchestrator,
    wall_sec: float,
    v0_dev: float,
    v0_holdout: float,
) -> dict[str, Any]:
    cands = orch.archive.list_candidates()
    loras = orch.archive.list_lora_adapters()
    best = orch.archive.best_candidate()
    v0 = orch.archive.get_candidate("cand_0000")

    mutated = [c for c in cands if c.generation > 0 and c.track == "scaffold"]
    smoke_ok = [
        c for c in mutated if c.status not in ("rejected_invalid", "rejected_smoke_failed", "error")
    ]
    valid_mut_rate = (len(smoke_ok) / len(mutated)) if mutated else None

    best_dev = candidate_primary_score(best) if best else 0.0
    best_anc = best.anchor_score if best and best.anchor_score is not None else 0.0
    best_ho = (
        best.holdout_score
        if best and best.holdout_score is not None
        else (v0.holdout_score if v0 and v0.holdout_score is not None else 0.0)
    )
    anchor_blocked = sum(1 for c in cands if c.status == "rejected_anchor_drop")

    return {
        "exp_id": exp_id,
        "label": label,
        "run_dir": str(run_dir),
        "best_candidate": best.candidate_id if best else "-",
        "best_track": best.track if best else "-",
        "best_lora": best.lora_adapter_id if best else None,
        "v0_dev": v0.dev_full_score if v0 else v0_dev,
        "best_dev": best_dev,
        "dev_gain": round(
            best_dev - (v0.dev_full_score if v0 and v0.dev_full_score is not None else v0_dev),
            4,
        ),
        "best_anchor": best_anc,
        "best_holdout": best_ho,
        "holdout_gain": round(best_ho - v0_holdout, 4),
        "overfit_gap": round((best_dev - v0_dev) - (best_ho - v0_holdout), 4),
        "valid_mutation_rate": round(valid_mut_rate, 4) if valid_mut_rate is not None else None,
        "anchor_blocked": anchor_blocked,
        "lora_samples": loras[-1].train_samples if loras else None,
        "lora_accepted": sum(1 for a in loras if a.accepted),
        "lora_loss": loras[-1].train_loss if loras else None,
        "total_tokens": orch.archive.total_tokens_spent(),
        "wall_sec": round(wall_sec, 1),
    }


def write_master_report(results: list[dict[str, Any]]) -> Path:
    (SUITE_ROOT / "ablation_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def _p(v: float | None) -> str:
        return f"{v:.1%}" if v is not None else "-"

    def _d(v: float | None) -> str:
        return f"{v:+.1%}" if v is not None else "-"

    lines = [
        "# RSI Agent MVP v1.2 — 6-Group Controlled Ablation Report (`Qwen3-8B` / `MATH-500`)",
        "",
        "## Experimental Setup",
        "- **Base Model**: `Qwen/Qwen3-8B` (BF16 on 1 × NVIDIA RTX 5090 32 GB)",
        "- **Dataset**: Stratified `MATH-500` (`Train=100 / Dev=20 / Anchor=16 / Holdout=30`; "
        "Exp-6 Unlabeled `Train=60 / Dev=20`)",
        "- **Upgrades Active**: Truncation telemetry, escape-free XML `<edit>` blocks + 1 retry, "
        "concise un-truncated SFT (`max_length=4096, lr=2e-5, 1 epoch, r=8`)",
        "",
        "## Master Comparison Table",
        "",
        "| Exp | Method | Best ID | Track / LoRA | Dev | Anchor | Holdout | "
        "Holdout Gain | Overfit Gap | Valid Mut | Anchor Blocks | LoRA (N/Loss) | Wall (s) |",
        "| :--- | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
    ]
    for r in results:
        track_str = r["best_track"] + (f" ({r['best_lora']})" if r["best_lora"] else "")
        lora_str = (
            f"{r['lora_samples']} / {r['lora_loss']:.3f}"
            if r["lora_loss"] is not None
            else "-"
        )
        lines.append(
            f"| **{r['exp_id']}** | {r['label']} | `{r['best_candidate']}` | `{track_str}` | "
            f"{_p(r['best_dev'])} | {_p(r['best_anchor'])} | **{_p(r['best_holdout'])}** | "
            f"{_d(r['holdout_gain'])} | {_d(r['overfit_gap'])} | {_p(r['valid_mutation_rate'])} | "
            f"{r['anchor_blocked']} | {lora_str} | {r['wall_sec']:.0f} |"
        )

    report_path = SUITE_ROOT / "ablation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    load_dotenv_file(REPO_ROOT / ".env")
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    prepare_ablation_splits(seed=42)

    results: list[dict[str, Any]] = []
    # Full 4096 sequence length so complete reasoning + \boxed{} is NEVER truncated!
    trainer = TRLLoRASFTTrainer(workspace_root=WORKSPACE_ROOT, max_length=4096)

    base_cfg = make_base_config("exp1_baseline_v0", LABELED_DS, max_generations=0)
    lifecycle = VLLMLifecycleManager(
        config=base_cfg,
        model_path_override=str(MODEL_DIR),
        gpu_memory_utilization=0.80,
        log_dir=SUITE_ROOT,
    )

    try:
        # --- Exp-1: Baseline v0 ---
        print("\n=== [Exp-1] Baseline v0 ===", flush=True)
        t0 = time.perf_counter()
        exp1_dir = SUITE_ROOT / "exp1_baseline_v0"
        lifecycle.ensure_serving(None)
        orch1 = EvolutionOrchestrator(
            config=base_cfg, run_dir=exp1_dir, lifecycle=lifecycle, trainer=trainer
        )
        orch1.run(max_generations=0)
        v0_cand = orch1.archive.get_candidate("cand_0000")
        assert v0_cand is not None
        v0_dev = v0_cand.dev_full_score or 0.0
        v0_ho = v0_cand.holdout_score or 0.0
        results.append(
            summarize_experiment(
                "Exp-1", "Baseline v0", exp1_dir, orch1, time.perf_counter() - t0, v0_dev, v0_ho
            )
        )
        print(
            f"[Exp-1] Done: dev={v0_dev:.1%}, anchor={v0_cand.anchor_score:.1%}, "
            f"holdout={v0_ho:.1%}",
            flush=True,
        )

        # --- Exp-2: Prompt-Only SI (4 gens) ---
        print("\n=== [Exp-2] Prompt-Only SI (prompts.py only, 4 gens) ===", flush=True)
        t0 = time.perf_counter()
        exp2_dir = SUITE_ROOT / "exp2_prompt_only_si"
        seed_v0_from_exp1(exp1_dir, exp2_dir)
        cfg2 = make_base_config(
            "exp2_prompt_only_si",
            LABELED_DS,
            max_generations=4,
            allowed_edit_files=["prompts.py"],
            weight_enabled=False,
        )
        orch2 = EvolutionOrchestrator(
            config=cfg2, run_dir=exp2_dir, lifecycle=lifecycle, trainer=trainer
        )
        orch2.run(max_generations=4)
        results.append(
            summarize_experiment(
                "Exp-2",
                "Prompt-Only SI (4g)",
                exp2_dir,
                orch2,
                time.perf_counter() - t0,
                v0_dev,
                v0_ho,
            )
        )
        print(
            f"[Exp-2] Done: {results[-1]['best_candidate']} dev={results[-1]['best_dev']:.1%} "
            f"ho={results[-1]['best_holdout']:.1%} mut={results[-1]['valid_mutation_rate']}",
            flush=True,
        )

        # --- Exp-3: Pure Scaffold RSI (4 gens) ---
        print("\n=== [Exp-3] Pure Scaffold RSI (all agent/*.py, 4 gens) ===", flush=True)
        t0 = time.perf_counter()
        exp3_dir = SUITE_ROOT / "exp3_pure_scaffold_rsi"
        seed_v0_from_exp1(exp1_dir, exp3_dir)
        cfg3 = make_base_config(
            "exp3_pure_scaffold_rsi",
            LABELED_DS,
            max_generations=4,
            allowed_edit_files=None,
            weight_enabled=False,
        )
        orch3 = EvolutionOrchestrator(
            config=cfg3, run_dir=exp3_dir, lifecycle=lifecycle, trainer=trainer
        )
        orch3.run(max_generations=4)
        results.append(
            summarize_experiment(
                "Exp-3",
                "Pure Scaffold RSI (4g)",
                exp3_dir,
                orch3,
                time.perf_counter() - t0,
                v0_dev,
                v0_ho,
            )
        )
        print(
            f"[Exp-3] Done: {results[-1]['best_candidate']} dev={results[-1]['best_dev']:.1%} "
            f"ho={results[-1]['best_holdout']:.1%} mut={results[-1]['valid_mutation_rate']}",
            flush=True,
        )

        # --- Exp-4: Pure Weight LoRA (Train=100, concise un-truncated SFT) ---
        print(
            "\n=== [Exp-4] Pure Weight LoRA (Train=100, lr=2e-5, 1ep, max_len=4096) ===",
            flush=True,
        )
        t0 = time.perf_counter()
        exp4_dir = SUITE_ROOT / "exp4_pure_weight_lora"
        seed_v0_from_exp1(exp1_dir, exp4_dir)
        cfg4 = make_base_config(
            "exp4_pure_weight_lora",
            LABELED_DS,
            max_generations=0,
            weight_enabled=True,
            trigger_every_n=1,
            min_train_samples=15,
            min_confidence=0.7,
        )
        lifecycle.config = cfg4
        orch4 = EvolutionOrchestrator(
            config=cfg4, run_dir=exp4_dir, lifecycle=lifecycle, trainer=trainer
        )
        orch4._run_weight_adaptation_round(generation=1)
        generate_summary_report(exp4_dir, write_file=True)
        results.append(
            summarize_experiment(
                "Exp-4",
                "Pure Weight LoRA",
                exp4_dir,
                orch4,
                time.perf_counter() - t0,
                v0_dev,
                v0_ho,
            )
        )
        print(
            f"[Exp-4] Done: {results[-1]['best_candidate']} dev={results[-1]['best_dev']:.1%} "
            f"ho={results[-1]['best_holdout']:.1%} lora_n={results[-1]['lora_samples']}",
            flush=True,
        )

        lifecycle._loaded_loras.clear()
        lifecycle.ensure_serving(None)

        # --- Exp-5: Full Dual-Track RSI (3 gens Scaffold + Track-B LoRA on Train=100) ---
        print("\n=== [Exp-5] Full Dual-Track RSI (Scaffold 3g + LoRA SFT) ===", flush=True)
        t0 = time.perf_counter()
        exp5_dir = SUITE_ROOT / "exp5_full_dual_track_rsi"
        seed_v0_from_exp1(exp1_dir, exp5_dir)
        cfg5 = make_base_config(
            "exp5_full_dual_track_rsi",
            LABELED_DS,
            max_generations=3,
            allowed_edit_files=None,
            weight_enabled=True,
            trigger_every_n=3,
            min_train_samples=15,
            min_confidence=0.7,
        )
        lifecycle.config = cfg5
        orch5 = EvolutionOrchestrator(
            config=cfg5, run_dir=exp5_dir, lifecycle=lifecycle, trainer=trainer
        )
        orch5.run(max_generations=3)
        results.append(
            summarize_experiment(
                "Exp-5",
                "Full Dual-Track RSI",
                exp5_dir,
                orch5,
                time.perf_counter() - t0,
                v0_dev,
                v0_ho,
            )
        )
        print(
            f"[Exp-5] Done: {results[-1]['best_candidate']} dev={results[-1]['best_dev']:.1%} "
            f"ho={results[-1]['best_holdout']:.1%}",
            flush=True,
        )

        lifecycle._loaded_loras.clear()
        lifecycle.ensure_serving(None)

        # --- Exp-6: Label-Free Consensus RSI (K=4, Train=60 unlabeled + Golden Anchor) ---
        print(
            "\n=== [Exp-6] Label-Free Consensus RSI (K=4, 3g + LoRA + Golden Anchor) ===",
            flush=True,
        )
        t0 = time.perf_counter()
        exp6_dir = SUITE_ROOT / "exp6_labelfree_consensus_rsi"
        cfg6 = make_base_config(
            "exp6_labelfree_consensus_rsi",
            UNLABELED_DS,
            verifier="consensus",
            num_samples=4,
            max_generations=3,
            allowed_edit_files=None,
            weight_enabled=True,
            trigger_every_n=3,
            min_train_samples=10,
            min_confidence=0.75,
        )
        lifecycle.config = cfg6
        orch6 = EvolutionOrchestrator(
            config=cfg6, run_dir=exp6_dir, lifecycle=lifecycle, trainer=trainer
        )
        orch6.run(max_generations=3)
        results.append(
            summarize_experiment(
                "Exp-6",
                "Label-Free Consensus RSI",
                exp6_dir,
                orch6,
                time.perf_counter() - t0,
                v0_dev,
                v0_ho,
            )
        )
        print(
            f"[Exp-6] Done: {results[-1]['best_candidate']} dev={results[-1]['best_dev']:.1%} "
            f"ho={results[-1]['best_holdout']:.1%}",
            flush=True,
        )

    finally:
        print("\n[ablation-v2] Shutting down vLLM server...", flush=True)
        lifecycle.shutdown()

    report_path = write_master_report(results)
    print(f"\n[ablation-v2] Complete! Master report: {report_path}\n", flush=True)
    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
