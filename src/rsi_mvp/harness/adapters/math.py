"""Symbolic math reasoning TaskAdapter (e.g. MATH-500, GSM8K)."""

from __future__ import annotations

import re
from typing import Any

from rsi_mvp.harness.adapters.base import register_adapter
from rsi_mvp.schemas import TaskInput, TaskSolution, TaskSpec, VerifyResult

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\$?([^\n$]+)\$?",
    flags=re.IGNORECASE,
)


def extract_boxed_answer(raw_text: str) -> str:
    """Extract the last balanced ``\\boxed{...}`` expression, with clean fallbacks."""
    cleaned = _THINK_RE.sub("", raw_text).strip()

    # Scan for all balanced \boxed{...} occurrences
    boxed_matches: list[str] = []
    marker = "\\boxed{"
    idx = 0
    while True:
        pos = cleaned.find(marker, idx)
        if pos == -1:
            break
        start = pos + len(marker)
        depth = 1
        cur = start
        while cur < len(cleaned) and depth > 0:
            ch = cleaned[cur]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            cur += 1
        if depth == 0:
            boxed_matches.append(cleaned[start : cur - 1].strip())
            idx = cur
        else:
            break

    if boxed_matches:
        return boxed_matches[-1]

    m = _FINAL_ANSWER_RE.findall(cleaned)
    if m:
        return m[-1].strip().rstrip(".")

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


class MathSymbolicAdapter:
    """Adapter for math reasoning benchmarks verified symbolically."""

    domain = "math"

    def to_task_input(self, spec: TaskSpec, num_samples: int = 1) -> TaskInput:
        return spec.to_task_input(num_samples=num_samples)

    def extract_answer(self, raw_text: str, spec: TaskSpec) -> str:
        _ = spec
        return extract_boxed_answer(raw_text)

    def format_for_sft(
        self, spec: TaskSpec, solution: TaskSolution, verify: VerifyResult
    ) -> dict[str, Any]:
        # When K > 1 samples exist (e.g. ConsensusVerifier), select the shortest complete
        # trace matching the consensus pseudo_label instead of blindly taking outputs[0].
        chosen = solution.outputs[0]
        if verify.pseudo_label and len(solution.outputs) > 1:
            matching = [
                o
                for o in solution.outputs
                if o.extracted_answer.strip() == verify.pseudo_label.strip()
                and not ("<think>" in o.raw_text and "</think>" not in o.raw_text)
            ]
            if matching:
                chosen = min(matching, key=lambda o: len(o.raw_text))

        assistant_text = chosen.raw_text.strip()
        if verify.pseudo_label and f"\\boxed{{{verify.pseudo_label}}}" not in assistant_text:
            assistant_text = f"{assistant_text}\n\nFinal answer: \\boxed{{{verify.pseudo_label}}}"
        return {
            "task_id": spec.task_id,
            "domain": self.domain,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert mathematician. Solve step by step and put the "
                        "final answer inside \\boxed{...}."
                    ),
                },
                {"role": "user", "content": spec.prompt},
                {"role": "assistant", "content": assistant_text},
            ],
        }


register_adapter("math", MathSymbolicAdapter)
