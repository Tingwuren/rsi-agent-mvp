#!/usr/bin/env python3
"""Prepare stratified Train/Dev/Holdout/Anchor datasets under /mnt/data0 workspace.

Supports:
  - mbpp_plus   (HuggingFace: evalplus/mbppplus -> 378 Python tasks)
  - math_500    (HuggingFace: HuggingFaceH4/MATH-500 -> 500 symbolic math tasks)
  - synthetic   (Offline deterministic mini code + math suites for zero-network smoke tests)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from rsi_mvp.config import check_storage_guard, load_dotenv_file
from rsi_mvp.schemas import SplitType, TaskSpec


def _write_splits(
    out_dir: Path,
    split_map: dict[SplitType, list[TaskSpec]],
    also_parquet: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, specs in split_map.items():
        jsonl_path = out_dir / f"{split_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for spec in specs:
                fh.write(spec.model_dump_json() + "\n")

        if also_parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq

            rows = []
            for spec in specs:
                d = spec.model_dump()
                d["payload"] = json.dumps(d["payload"], ensure_ascii=False)
                d["ground_truth"] = (
                    json.dumps(d["ground_truth"], ensure_ascii=False)
                    if d["ground_truth"] is not None
                    else ""
                )
                rows.append(d)
            pq.write_table(pa.Table.from_pylist(rows), out_dir / f"{split_name}.parquet")

    counts = {k: len(v) for k, v in split_map.items()}
    print(f"[prepare] Wrote {out_dir}: {counts}")


def _stratified_partition(
    items: list[dict[str, Any]],
    counts: dict[SplitType, int],
    seed: int = 42,
) -> dict[SplitType, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_cat[str(item.get("category", "general"))].append(item)

    interleaved: list[dict[str, Any]] = []
    for cat in sorted(by_cat):
        rng.shuffle(by_cat[cat])
    while any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat]:
                interleaved.append(by_cat[cat].pop())

    result: dict[SplitType, list[dict[str, Any]]] = {}
    cursor = 0
    for split_name in ("anchor", "holdout", "dev", "train"):
        n = counts.get(split_name, 0)
        result[split_name] = interleaved[cursor : cursor + n]
        cursor += n
    return result


def prepare_mbpp_plus(out_dir: Path, seed: int = 42) -> None:
    from datasets import load_dataset

    ds = load_dataset("evalplus/mbppplus", split="test")
    raw_items: list[dict[str, Any]] = []
    for row in ds:
        task_id = str(row["task_id"]).replace("/", "_")
        base_prompt = str(row["prompt"]).strip()
        test_list = row.get("test_list") or []
        test_imports = row.get("test_imports") or []
        # Follow official MBPP/EvalPlus prompt convention: expose the first public
        # example assertion so the required function name & I/O signature are unambiguous,
        # while keeping the remaining assertions hidden in ground_truth.
        if test_list:
            prompt = (
                f"{base_prompt}\n\n"
                f"Your solution must define the function matching this example:\n"
                f"```python\n{test_list[0]}\n```"
            )
        else:
            prompt = base_prompt
        test_lines = [
            "from solution import *",
            *test_imports,
            "def test_mbpp_cases():",
        ]
        for assert_stmt in test_list:
            test_lines.append(f"    {assert_stmt}")
        raw_items.append(
            {
                "task_id": task_id,
                "domain": "code",
                "category": "python_stdlib",
                "prompt": prompt,
                "payload": {"entry_file": "solution.py"},
                "ground_truth": {"test_code": "\n".join(test_lines) + "\n"},
                "timeout_sec": 15,
            }
        )

    total = len(raw_items)
    counts: dict[SplitType, int] = {
        "anchor": min(40, total // 10),
        "holdout": min(88, total // 4),
        "dev": min(70, total // 5),
        "train": max(0, total - min(40, total // 10) - min(88, total // 4) - min(70, total // 5)),
    }
    part = _stratified_partition(raw_items, counts=counts, seed=seed)
    split_specs: dict[SplitType, list[TaskSpec]] = {
        s: [TaskSpec(split=s, **item) for item in rows] for s, rows in part.items()
    }
    _write_splits(out_dir, split_specs)


def prepare_math_500(out_dir: Path, seed: int = 42, max_total: int = 240) -> None:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    raw_items: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        subject = str(row.get("subject", "math")).lower().replace(" ", "_")
        level = row.get("level", 3)
        raw_items.append(
            {
                "task_id": str(row.get("unique_id", f"math500_{idx:04d}")).replace("/", "_"),
                "domain": "math",
                "category": f"{subject}_L{level}",
                "prompt": str(row["problem"]).strip(),
                "payload": {"subject": subject, "level": level},
                "ground_truth": {"answer": str(row["answer"]).strip()},
                "timeout_sec": 15,
            }
        )

    cap = min(max_total, len(raw_items))
    counts: dict[SplitType, int] = {
        "anchor": min(40, cap // 6),
        "holdout": min(50, cap // 4),
        "dev": min(50, cap // 4),
        "train": max(0, cap - min(40, cap // 6) - min(50, cap // 4) - min(50, cap // 4)),
    }
    part = _stratified_partition(raw_items, counts=counts, seed=seed)
    split_specs: dict[SplitType, list[TaskSpec]] = {
        s: [TaskSpec(split=s, **item) for item in rows] for s, rows in part.items()
    }
    _write_splits(out_dir, split_specs)


def prepare_bigcodebench(out_dir: Path, seed: int = 42, max_total: int = 280) -> None:
    """Prepare a stratified 280-task BigCodeBench-Instruct subset (120/60/60/40)."""
    from datasets import load_dataset

    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
    raw_items: list[dict[str, Any]] = []
    for row in ds:
        task_id = str(row["task_id"]).replace("/", "_")
        prompt = str(row.get("instruct_prompt") or row.get("complete_prompt") or "").strip()
        libs = row.get("libs") or []
        category = str(libs[0]).lower() if isinstance(libs, list) and libs else "stdlib"
        test_code = "from solution import *\n" + str(row["test"])
        raw_items.append(
            {
                "task_id": task_id,
                "domain": "code",
                "category": category,
                "prompt": prompt,
                "payload": {
                    "entry_file": "solution.py",
                    "entry_point": str(row.get("entry_point", "task_func")),
                    "libs": libs,
                },
                "ground_truth": {"test_code": test_code},
                "timeout_sec": 25,
            }
        )

    cap = min(max_total, len(raw_items))
    counts: dict[SplitType, int] = {
        "anchor": min(40, cap // 7),
        "holdout": min(60, cap // 4),
        "dev": min(60, cap // 4),
        "train": max(0, cap - min(40, cap // 7) - min(60, cap // 4) - min(60, cap // 4)),
    }
    part = _stratified_partition(raw_items, counts=counts, seed=seed)
    split_specs: dict[SplitType, list[TaskSpec]] = {
        s: [TaskSpec(split=s, **item) for item in rows] for s, rows in part.items()
    }
    _write_splits(out_dir, split_specs)


def prepare_synthetic(workspace_root: Path) -> None:
    """Create tiny offline synthetic code + math suites for instant local verification."""
    code_items = [
        (
            f"syn_code_{i:02d}",
            "arithmetic" if i % 2 == 0 else "strings",
            (
                f"Write a Python function `solve_fn(x)` that returns `x + {i}`."
                if i % 2 == 0
                else f"Write a Python function `solve_fn(s)` that returns `s.upper() + '{i}'`."
            ),
            (
                f"from solution import solve_fn\n"
                f"def test_case():\n    assert solve_fn(10) == {10 + i}\n"
                if i % 2 == 0
                else f"from solution import solve_fn\n"
                f"def test_case():\n    assert solve_fn('ab') == 'AB{i}'\n"
            ),
        )
        for i in range(16)
    ]
    code_splits: dict[SplitType, list[TaskSpec]] = {
        "train": [],
        "dev": [],
        "holdout": [],
        "anchor": [],
    }
    split_order: list[SplitType] = ["train", "dev", "holdout", "anchor"]
    for idx, (tid, cat, prompt, test_code) in enumerate(code_items):
        sp = split_order[idx % 4]
        code_splits[sp].append(
            TaskSpec(
                task_id=tid,
                domain="code",
                split=sp,
                category=cat,
                prompt=prompt,
                payload={"entry_file": "solution.py"},
                ground_truth={"test_code": test_code},
            )
        )
    _write_splits(workspace_root / "datasets" / "synthetic_code", code_splits)

    math_splits: dict[SplitType, list[TaskSpec]] = {
        "train": [],
        "dev": [],
        "holdout": [],
        "anchor": [],
    }
    for i in range(16):
        sp = split_order[i % 4]
        a, b = i + 2, i + 5
        math_splits[sp].append(
            TaskSpec(
                task_id=f"syn_math_{i:02d}",
                domain="math",
                split=sp,
                category="algebra",
                prompt=f"Compute ({a} * {b}) + {i}. Put your final answer in \\boxed{{...}}.",
                ground_truth={"answer": str(a * b + i)},
            )
        )
    _write_splits(workspace_root / "datasets" / "synthetic_math", math_splits)


def main() -> int:
    load_dotenv_file(Path(".env"))
    parser = argparse.ArgumentParser(description="Prepare RSI Agent MVP benchmark splits")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mbpp_plus", "math_500", "bigcodebench", "synthetic", "all"],
    )
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=os.environ.get("RSI_WORKSPACE_ROOT", str(Path.cwd() / ".workspace")),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ws = Path(args.workspace_root)
    check_storage_guard(ws, min_free_gb=5.0)
    os.environ.setdefault("HF_HOME", str(ws / "hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ws / "hf_cache" / "datasets"))

    if args.dataset in ("synthetic", "all"):
        prepare_synthetic(ws)
    if args.dataset in ("mbpp_plus", "all"):
        prepare_mbpp_plus(ws / "datasets" / "mbpp_plus", seed=args.seed)
    if args.dataset in ("math_500", "all"):
        prepare_math_500(ws / "datasets" / "math_500", seed=args.seed)
    if args.dataset == "bigcodebench":
        prepare_bigcodebench(ws / "datasets" / "bigcodebench_280", seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
