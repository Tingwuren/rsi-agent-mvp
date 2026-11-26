"""Unit tests for DatasetProvider (JSONL & Parquet) and TaskAdapters."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rsi_mvp.harness.adapters import get_adapter, list_adapters
from rsi_mvp.harness.data import HFParquetDatasetProvider, JSONLDatasetProvider
from rsi_mvp.schemas import CandidateOutput, TaskSolution, TaskSpec, VerifyResult


def _make_sample_tasks(n: int = 30) -> list[TaskSpec]:
    cats = ["cat_a", "cat_b", "cat_c"]
    return [
        TaskSpec(
            task_id=f"t_{i:03d}",
            domain="code",
            split="dev",
            category=cats[i % 3],
            prompt=f"Solve {i}",
            ground_truth={"test_code": "assert True"},
        )
        for i in range(n)
    ]


def test_jsonl_provider_stratified_sampling_and_batches(tmp_path: Path) -> None:
    tasks = _make_sample_tasks(30)
    (tmp_path / "dev.jsonl").write_text(
        "\n".join(t.model_dump_json() for t in tasks) + "\n",
        encoding="utf-8",
    )
    provider = JSONLDatasetProvider(tmp_path)

    sub1 = provider.get_split("dev", max_samples=9, stratified=True, seed=123)
    sub2 = provider.get_split("dev", max_samples=9, stratified=True, seed=123)
    assert [t.task_id for t in sub1] == [t.task_id for t in sub2]
    assert Counter(t.category for t in sub1) == {"cat_a": 3, "cat_b": 3, "cat_c": 3}

    batches = list(provider.iter_batches("dev", batch_size=8))
    assert [len(b) for b in batches] == [8, 8, 8, 6]


def test_parquet_provider_stratified_and_streaming_batches(tmp_path: Path) -> None:
    tasks = _make_sample_tasks(24)
    rows = []
    for t in tasks:
        d = t.model_dump()
        d["payload"] = json.dumps(d["payload"])
        d["ground_truth"] = json.dumps(d["ground_truth"])
        rows.append(d)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "dev.parquet")

    provider = HFParquetDatasetProvider(tmp_path)
    sampled = provider.get_split("dev", max_samples=6, stratified=True, seed=7)
    assert Counter(t.category for t in sampled) == {"cat_a": 2, "cat_b": 2, "cat_c": 2}

    batches = list(provider.iter_batches("dev", batch_size=10))
    assert [len(b) for b in batches] == [10, 10, 4]


def test_adapters_extract_and_format_sft() -> None:
    assert {"code", "math", "qa"} <= set(list_adapters())

    code_adapter = get_adapter("code")
    code_spec = TaskSpec(
        task_id="c1",
        domain="code",
        split="train",
        prompt="Return x*2",
        ground_truth={"test_code": "assert True"},
    )
    raw_code = "<think>hmm</think>\n```python\ndef f(x):\n    return x * 2\n```"
    extracted_code = code_adapter.extract_answer(raw_code, code_spec)
    assert extracted_code == "def f(x):\n    return x * 2"

    sol = TaskSolution(
        task_id="c1",
        outputs=[CandidateOutput(raw_text=raw_code, extracted_answer=extracted_code)],
    )
    ver = VerifyResult(task_id="c1", passed=True, score=1.0)
    sft = code_adapter.format_for_sft(code_spec, sol, ver)
    assert len(sft["messages"]) == 3
    assert "return x * 2" in sft["messages"][2]["content"]

    math_adapter = get_adapter("math")
    math_spec = TaskSpec(task_id="m1", domain="math", split="dev", prompt="1/2 + 1/2")
    raw_math = "<think>0.5+0.5=1</think>Therefore the answer is \\boxed{\\frac{2}{2}}."
    assert math_adapter.extract_answer(raw_math, math_spec) == "\\frac{2}{2}"
