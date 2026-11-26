"""DatasetProvider abstractions supporting JSONL, Parquet/Arrow, and stratified sampling."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from rsi_mvp.schemas import DatasetConfig, SplitType, TaskSpec


class DatasetProvider(Protocol):
    """Unified dataset access protocol for small benchmarks and large datasets."""

    def get_split(
        self,
        split: SplitType,
        max_samples: int | None = None,
        stratified: bool = True,
        seed: int = 42,
    ) -> list[TaskSpec]: ...

    def iter_batches(self, split: SplitType, batch_size: int) -> Iterator[list[TaskSpec]]: ...


def _stratified_sample(tasks: list[TaskSpec], max_samples: int, seed: int) -> list[TaskSpec]:
    """Deterministically sample up to ``max_samples`` preserving ``category`` proportions."""
    if max_samples >= len(tasks):
        return list(tasks)

    rng = random.Random(seed)
    buckets: dict[str, list[TaskSpec]] = defaultdict(list)
    for task in tasks:
        buckets[task.category].append(task)

    for cat in sorted(buckets):
        rng.shuffle(buckets[cat])

    selected: list[TaskSpec] = []
    # Proportional floor allocation first
    remainders: list[tuple[float, str]] = []
    total = len(tasks)
    for cat in sorted(buckets):
        exact = len(buckets[cat]) * max_samples / total
        take = min(len(buckets[cat]), int(exact))
        selected.extend(buckets[cat][:take])
        buckets[cat] = buckets[cat][take:]
        remainders.append((exact - take, cat))

    # Fill leftover slots by largest fractional remainder
    remainders.sort(key=lambda pair: (-pair[0], pair[1]))
    for _, cat in remainders:
        if len(selected) >= max_samples:
            break
        if buckets[cat]:
            selected.append(buckets[cat].pop(0))

    # Deterministic final fill if any slot still remains
    if len(selected) < max_samples:
        leftover: list[TaskSpec] = []
        for cat in sorted(buckets):
            leftover.extend(buckets[cat])
        rng.shuffle(leftover)
        selected.extend(leftover[: max_samples - len(selected)])

    rng.shuffle(selected)
    return selected[:max_samples]


def _coerce_task_spec(row: dict[str, Any], split: SplitType) -> TaskSpec:
    data = dict(row)
    data.setdefault("split", split)
    for json_field in ("payload", "ground_truth"):
        val = data.get(json_field)
        if isinstance(val, str):
            stripped = val.strip()
            data[json_field] = json.loads(stripped) if stripped else None
    return TaskSpec.model_validate(data)


class JSONLDatasetProvider:
    """Loads ``<root>/{train,dev,holdout,anchor}.jsonl`` task splits."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)

    def _split_file(self, split: SplitType) -> Path:
        path = self.root_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Split file not found: {path}")
        return path

    def get_split(
        self,
        split: SplitType,
        max_samples: int | None = None,
        stratified: bool = True,
        seed: int = 42,
    ) -> list[TaskSpec]:
        path = self._split_file(split)
        tasks: list[TaskSpec] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tasks.append(_coerce_task_spec(json.loads(line), split))

        if max_samples is None or max_samples >= len(tasks):
            return tasks
        if stratified:
            return _stratified_sample(tasks, max_samples=max_samples, seed=seed)
        rng = random.Random(seed)
        copied = list(tasks)
        rng.shuffle(copied)
        return copied[:max_samples]

    def iter_batches(self, split: SplitType, batch_size: int) -> Iterator[list[TaskSpec]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        path = self._split_file(split)
        batch: list[TaskSpec] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                batch.append(_coerce_task_spec(json.loads(line), split))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


class HFParquetDatasetProvider:
    """Memory-mapped Parquet/Arrow provider for medium-to-large datasets via ``pyarrow``."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)

    def _split_file(self, split: SplitType) -> Path:
        path = self.root_dir / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Parquet split file not found: {path}")
        return path

    def get_split(
        self,
        split: SplitType,
        max_samples: int | None = None,
        stratified: bool = True,
        seed: int = 42,
    ) -> list[TaskSpec]:
        import pyarrow.parquet as pq

        path = self._split_file(split)
        table = pq.read_table(path)
        rows = table.to_pylist()
        tasks = [_coerce_task_spec(r, split) for r in rows]

        if max_samples is None or max_samples >= len(tasks):
            return tasks
        if stratified:
            return _stratified_sample(tasks, max_samples=max_samples, seed=seed)
        rng = random.Random(seed)
        rng.shuffle(tasks)
        return tasks[:max_samples]

    def iter_batches(self, split: SplitType, batch_size: int) -> Iterator[list[TaskSpec]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        import pyarrow.parquet as pq

        path = self._split_file(split)
        pf = pq.ParquetFile(path)
        for record_batch in pf.iter_batches(batch_size=batch_size):
            yield [_coerce_task_spec(r, split) for r in record_batch.to_pylist()]


def create_dataset_provider(cfg: DatasetConfig) -> DatasetProvider:
    """Instantiate the configured ``DatasetProvider``."""
    root = Path(cfg.path)
    if cfg.provider == "jsonl":
        return JSONLDatasetProvider(root)
    if cfg.provider in ("hf_local_parquet", "hf_streaming"):
        return HFParquetDatasetProvider(root)
    raise ValueError(f"Unsupported dataset provider: {cfg.provider}")
