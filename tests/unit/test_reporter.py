"""Unit tests for Markdown experiment reporter and ``rsi report`` CLI."""

from __future__ import annotations

import shutil
from pathlib import Path

from rsi_mvp.cli import main as cli_main
from rsi_mvp.harness.archive import ArchiveStore
from rsi_mvp.harness.reporter import generate_summary_report
from rsi_mvp.schemas import CandidateRecord, LoRAAdapterRecord


def test_generate_summary_report_with_overfit_warning_diff_and_cli(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp_report"
    store = ArchiveStore(run_dir)

    seed = Path("src/rsi_mvp/agent")
    shutil.copytree(seed, run_dir / "snapshots" / "cand_0000" / "agent")
    shutil.copytree(seed, run_dir / "snapshots" / "cand_0001" / "agent")
    prompts_v1 = run_dir / "snapshots" / "cand_0001" / "agent" / "prompts.py"
    prompts_v1.write_text(
        prompts_v1.read_text(encoding="utf-8") + "\n# RSI_DIFF_MARKER\n",
        encoding="utf-8",
    )

    store.add_candidate(
        CandidateRecord(
            candidate_id="cand_0000",
            generation=0,
            track="bootstrap",
            status="accepted",
            snapshot_dir="snapshots/cand_0000",
            hypothesis="Baseline v0",
            dev_mini_score=0.50,
            dev_full_score=0.50,
            anchor_score=0.70,
            holdout_score=0.50,
            output_entropy=0.82,
            total_tokens=1000,
        )
    )
    store.add_candidate(
        CandidateRecord(
            candidate_id="cand_0001",
            generation=1,
            parent_id="cand_0000",
            track="scaffold",
            status="accepted",
            snapshot_dir="snapshots/cand_0001",
            hypothesis="Overfit dev prompt",
            dev_mini_score=0.90,
            dev_full_score=0.90,
            anchor_score=0.70,
            holdout_score=0.55,  # Dev +40%, Holdout +5% -> Overfit Gap +35% (>15% warning!)
            output_entropy=0.40,
            total_tokens=1100,
        )
    )
    store.add_candidate(
        CandidateRecord(
            candidate_id="cand_0002",
            generation=2,
            parent_id="cand_0001",
            track="weight_lora",
            lora_adapter_id="lora_0001",
            status="rejected_anchor_drop",
            snapshot_dir="snapshots/cand_0001",
            hypothesis="Drifted LoRA",
            dev_full_score=0.95,
            anchor_score=0.55,
            total_tokens=900,
        )
    )
    store.add_lora_adapter(
        LoRAAdapterRecord(
            adapter_id="lora_0001",
            base_model="Qwen/Qwen3-8B",
            adapter_path="/mnt/data0/adapters/lora_0001",
            train_samples=64,
            train_loss=0.33,
            anchor_delta=-0.15,
            accepted=False,
        )
    )

    md = generate_summary_report(run_dir, write_file=True)
    assert (run_dir / "summary.md").is_file()
    assert "Overfitting Alert" in md
    assert "Anchor Gatekeeper Active" in md
    assert "RSI_DIFF_MARKER" in md
    assert "lora_0001" in md

    assert cli_main(["report", "--run", str(run_dir)]) == 0
