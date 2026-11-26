"""Generic structured QA TaskAdapter."""

from __future__ import annotations

import re
from typing import Any

from rsi_mvp.harness.adapters.base import register_adapter
from rsi_mvp.schemas import TaskInput, TaskSolution, TaskSpec, VerifyResult

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


class GenericQAAdapter:
    """Fallback adapter for general QA, classification, or instruction tasks."""

    domain = "qa"

    def to_task_input(self, spec: TaskSpec, num_samples: int = 1) -> TaskInput:
        return spec.to_task_input(num_samples=num_samples)

    def extract_answer(self, raw_text: str, spec: TaskSpec) -> str:
        _ = spec
        return _THINK_RE.sub("", raw_text).strip()

    def format_for_sft(
        self, spec: TaskSpec, solution: TaskSolution, verify: VerifyResult
    ) -> dict[str, Any]:
        ans = verify.pseudo_label or solution.outputs[0].extracted_answer
        return {
            "task_id": spec.task_id,
            "domain": self.domain,
            "messages": [
                {"role": "system", "content": "Provide a clear, accurate, concise answer."},
                {"role": "user", "content": spec.prompt},
                {"role": "assistant", "content": ans},
            ],
        }


register_adapter("qa", GenericQAAdapter)
