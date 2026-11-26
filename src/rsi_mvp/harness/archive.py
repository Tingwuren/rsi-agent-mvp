"""SQLite dual-track ArchiveStore and evolutionary parent selection policies."""

from __future__ import annotations

import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Any, Protocol

from rsi_mvp.schemas import CandidateRecord, LoRAAdapterRecord, utc_now_iso

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL,
    parent_id TEXT,
    track TEXT NOT NULL,
    lora_adapter_id TEXT,
    status TEXT NOT NULL,
    snapshot_dir TEXT NOT NULL,
    diff_path TEXT,
    hypothesis TEXT NOT NULL DEFAULT '',
    dev_mini_score REAL,
    dev_full_score REAL,
    anchor_score REAL,
    holdout_score REAL,
    output_entropy REAL,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    selection_count INTEGER NOT NULL DEFAULT 0,
    failure_reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lora_adapters (
    adapter_id TEXT PRIMARY KEY,
    base_model TEXT NOT NULL,
    parent_adapter_id TEXT,
    adapter_path TEXT NOT NULL,
    train_samples INTEGER NOT NULL,
    method TEXT NOT NULL,
    train_loss REAL,
    anchor_delta REAL,
    accepted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


def candidate_primary_score(cand: CandidateRecord) -> float:
    """Return the primary dev ranking score for a candidate."""
    if cand.dev_full_score is not None:
        return cand.dev_full_score
    if cand.dev_mini_score is not None:
        return cand.dev_mini_score
    return 0.0


class ArchiveStore:
    """Persistent SQLite store for dual-track (scaffold + LoRA) RSI lineage."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.run_dir / "archive.db"
        self.events_path = self.run_dir / "events.jsonl"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> CandidateRecord:
        return CandidateRecord.model_validate(dict(row))

    @staticmethod
    def _row_to_lora(row: sqlite3.Row) -> LoRAAdapterRecord:
        data = dict(row)
        data["accepted"] = bool(data["accepted"])
        return LoRAAdapterRecord.model_validate(data)

    def add_candidate(self, record: CandidateRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO candidates (
                    candidate_id, generation, parent_id, track, lora_adapter_id,
                    status, snapshot_dir, diff_path, hypothesis,
                    dev_mini_score, dev_full_score, anchor_score, holdout_score,
                    output_entropy, total_tokens, selection_count, failure_reason, created_at
                ) VALUES (
                    :candidate_id, :generation, :parent_id, :track, :lora_adapter_id,
                    :status, :snapshot_dir, :diff_path, :hypothesis,
                    :dev_mini_score, :dev_full_score, :anchor_score, :holdout_score,
                    :output_entropy, :total_tokens, :selection_count, :failure_reason, :created_at
                )
                """,
                record.model_dump(),
            )

    def get_candidate(self, candidate_id: str) -> CandidateRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._row_to_candidate(row) if row is not None else None

    def list_candidates(self) -> list[CandidateRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates ORDER BY generation ASC, candidate_id ASC"
            ).fetchall()
        return [self._row_to_candidate(r) for r in rows]

    def accepted_candidates(self) -> list[CandidateRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE status = 'accepted' "
                "ORDER BY generation ASC, candidate_id ASC"
            ).fetchall()
        return [self._row_to_candidate(r) for r in rows]

    def best_candidate(self) -> CandidateRecord | None:
        accepted = self.accepted_candidates()
        if not accepted:
            return None
        return max(
            accepted,
            key=lambda c: (candidate_primary_score(c), -c.total_tokens, -c.generation),
        )

    def increment_selection_count(self, candidate_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE candidates SET selection_count = selection_count + 1 "
                "WHERE candidate_id = ?",
                (candidate_id,),
            )

    def max_generation(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(generation) AS max_g FROM candidates").fetchone()
        if row is None or row["max_g"] is None:
            return -1
        return int(row["max_g"])

    def next_candidate_id(self) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM candidates").fetchone()
        idx = int(row["cnt"]) if row is not None else 0
        while self.get_candidate(f"cand_{idx:04d}") is not None:
            idx += 1
        return f"cand_{idx:04d}"

    def total_tokens_spent(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) AS s FROM candidates"
            ).fetchone()
        return int(row["s"]) if row is not None else 0

    def add_lora_adapter(self, record: LoRAAdapterRecord) -> None:
        payload = record.model_dump()
        payload["accepted"] = 1 if record.accepted else 0
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO lora_adapters (
                    adapter_id, base_model, parent_adapter_id, adapter_path,
                    train_samples, method, train_loss, anchor_delta, accepted, created_at
                ) VALUES (
                    :adapter_id, :base_model, :parent_adapter_id, :adapter_path,
                    :train_samples, :method, :train_loss, :anchor_delta, :accepted, :created_at
                )
                """,
                payload,
            )

    def list_lora_adapters(self) -> list[LoRAAdapterRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM lora_adapters ORDER BY created_at ASC, adapter_id ASC"
            ).fetchall()
        return [self._row_to_lora(r) for r in rows]

    def record_event(self, event_type: str, data: dict[str, Any]) -> None:
        entry = {"ts": utc_now_iso(), "event": event_type, **data}
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


class SelectionPolicy(Protocol):
    """Protocol for parent selection and candidate acceptance decisions."""

    name: str

    def select_parent(
        self, accepted: list[CandidateRecord], rng: random.Random
    ) -> CandidateRecord: ...

    def should_accept(
        self,
        new_score: float,
        new_tokens: int,
        best: CandidateRecord | None,
    ) -> bool: ...


class HillClimbingPolicy:
    """Strict greedy hill climbing: always mutates best parent and requires improvement."""

    name = "hill_climbing"

    def select_parent(
        self, accepted: list[CandidateRecord], rng: random.Random
    ) -> CandidateRecord:
        _ = rng
        if not accepted:
            raise ValueError("Cannot select parent from an empty accepted pool")
        return max(
            accepted,
            key=lambda c: (candidate_primary_score(c), -c.total_tokens, -c.generation),
        )

    def should_accept(
        self,
        new_score: float,
        new_tokens: int,
        best: CandidateRecord | None,
    ) -> bool:
        if best is None:
            return True
        best_score = candidate_primary_score(best)
        if new_score > best_score + 1e-9:
            return True
        # Tie on score: accept if token cost dropped by at least 10%
        if abs(new_score - best_score) <= 1e-9 and best.total_tokens > 0:
            return new_tokens <= int(best.total_tokens * 0.9)
        return False


class EvolutionaryArchivePolicy:
    """Open-ended DGM-style archive policy balancing performance and exploration diversity."""

    name = "evolutionary_archive"

    def __init__(
        self,
        score_sharpness: float = 5.0,
        diversity_penalty: float = 0.5,
        max_drop_tolerance: float = 0.25,
    ) -> None:
        self.score_sharpness = score_sharpness
        self.diversity_penalty = diversity_penalty
        self.max_drop_tolerance = max_drop_tolerance

    def compute_weights(self, accepted: list[CandidateRecord]) -> list[float]:
        if not accepted:
            return []
        scores = [candidate_primary_score(c) for c in accepted]
        mean_score = sum(scores) / len(scores)
        weights: list[float] = []
        for cand, score in zip(accepted, scores, strict=True):
            z = max(-20.0, min(20.0, self.score_sharpness * (score - mean_score)))
            sig = 1.0 / (1.0 + math.exp(-z))
            w = sig / (1.0 + self.diversity_penalty * cand.selection_count)
            weights.append(max(w, 1e-6))
        return weights

    def select_parent(
        self, accepted: list[CandidateRecord], rng: random.Random
    ) -> CandidateRecord:
        if not accepted:
            raise ValueError("Cannot select parent from an empty accepted pool")
        weights = self.compute_weights(accepted)
        return rng.choices(accepted, weights=weights, k=1)[0]

    def should_accept(
        self,
        new_score: float,
        new_tokens: int,
        best: CandidateRecord | None,
    ) -> bool:
        _ = new_tokens
        if best is None:
            return True
        best_score = candidate_primary_score(best)
        return new_score >= (best_score - self.max_drop_tolerance) - 1e-9


def create_selection_policy(name: str) -> SelectionPolicy:
    if name == "hill_climbing":
        return HillClimbingPolicy()
    if name == "evolutionary_archive":
        return EvolutionaryArchivePolicy()
    raise ValueError(f"Unsupported selection_policy: {name!r}")
