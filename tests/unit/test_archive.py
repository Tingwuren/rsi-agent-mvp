"""Unit tests for SQLite ArchiveStore and selection policies (TC-ARCH-01 / TC-ARCH-02)."""

from __future__ import annotations

import random
from pathlib import Path

from rsi_mvp.harness.archive import (
    ArchiveStore,
    EvolutionaryArchivePolicy,
    HillClimbingPolicy,
    create_selection_policy,
)
from rsi_mvp.schemas import CandidateRecord, LoRAAdapterRecord


def _cand(
    cid: str,
    gen: int,
    score: float,
    tokens: int = 1000,
    sel: int = 0,
    status: str = "accepted",
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=cid,
        generation=gen,
        parent_id=None if gen == 0 else "cand_0000",
        track="bootstrap" if gen == 0 else "scaffold",
        status=status,  # type: ignore[arg-type]
        snapshot_dir=f"snapshots/{cid}",
        dev_mini_score=score,
        dev_full_score=score,
        total_tokens=tokens,
        selection_count=sel,
    )


def test_archive_store_persists_candidates_and_lora_adapters(tmp_path: Path) -> None:
    store = ArchiveStore(tmp_path / "run1")
    c0 = _cand("cand_0000", 0, 0.50, tokens=2000)
    c1 = _cand("cand_0001", 1, 0.75, tokens=1800)
    c2 = _cand("cand_0002", 2, 0.20, status="rejected_invalid")
    store.add_candidate(c0)
    store.add_candidate(c1)
    store.add_candidate(c2)

    assert store.max_generation() == 2
    assert store.next_candidate_id() == "cand_0003"
    assert len(store.accepted_candidates()) == 2
    best = store.best_candidate()
    assert best is not None and best.candidate_id == "cand_0001"

    store.increment_selection_count("cand_0001")
    reloaded = store.get_candidate("cand_0001")
    assert reloaded is not None and reloaded.selection_count == 1

    lora = LoRAAdapterRecord(
        adapter_id="lora_0001",
        base_model="Qwen/Qwen3-8B",
        adapter_path="/mnt/data0/adapters/lora_0001",
        train_samples=100,
        accepted=True,
    )
    store.add_lora_adapter(lora)
    adapters = store.list_lora_adapters()
    assert len(adapters) == 1 and adapters[0].accepted is True


def test_tc_arch_01_hill_climbing_policy() -> None:
    policy = HillClimbingPolicy()
    best = _cand("cand_0001", 1, 0.60, tokens=1000)

    # Same score without >=10% token reduction -> rejected
    assert policy.should_accept(0.60, 950, best) is False
    # Same score with >=10% token reduction -> accepted
    assert policy.should_accept(0.60, 850, best) is True
    # Higher score -> accepted
    assert policy.should_accept(0.65, 1500, best) is True

    chosen = policy.select_parent([_cand("cand_0000", 0, 0.4), best], random.Random(0))
    assert chosen.candidate_id == "cand_0001"


def test_tc_arch_02_evolutionary_archive_diversity_penalty() -> None:
    policy = EvolutionaryArchivePolicy(score_sharpness=5.0, diversity_penalty=0.5)
    c_high_fresh = _cand("cand_0001", 1, 0.70, sel=0)
    c_mid_fresh = _cand("cand_0000", 0, 0.55, sel=0)
    w_before = policy.compute_weights([c_mid_fresh, c_high_fresh])
    p_mid_before = w_before[0] / sum(w_before)

    # After high candidate is selected many times, diversity penalty boosts alternative branch
    c_high_overused = c_high_fresh.model_copy(update={"selection_count": 10})
    w_after = policy.compute_weights([c_mid_fresh, c_high_overused])
    p_mid_after = w_after[0] / sum(w_after)
    assert p_mid_after > 0.75
    assert p_mid_after > p_mid_before * 1.8

    # Factory check
    assert create_selection_policy("hill_climbing").name == "hill_climbing"
    assert create_selection_policy("evolutionary_archive").name == "evolutionary_archive"
