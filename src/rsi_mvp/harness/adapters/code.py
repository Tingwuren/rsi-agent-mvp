"""Python code generation / bugfix TaskAdapter."""

from __future__ import annotations

import re
from typing import Any

from rsi_mvp.harness.adapters.base import register_adapter
from rsi_mvp.schemas import TaskInput, TaskSolution, TaskSpec, VerifyResult

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def extract_python_code(raw_text: str) -> str:
    """Extract Python source code from a model response (stripping ``<think>`` blocks)."""
    cleaned = _THINK_RE.sub("", raw_text).strip()
    matches = _CODE_BLOCK_RE.findall(cleaned)
    if matches:
        return matches[-1].strip()
    return cleaned


class CodeUnitTestAdapter:
    """Adapter for Python function/module coding and bugfix benchmarks."""

    domain = "code"

    def to_task_input(self, spec: TaskSpec, num_samples: int = 1) -> TaskInput:
        return spec.to_task_input(num_samples=num_samples)

    def extract_answer(self, raw_text: str, spec: TaskSpec) -> str:
        _ = spec
        return extract_python_code(raw_text)

    def format_for_sft(
        self, spec: TaskSpec, solution: TaskSolution, verify: VerifyResult
    ) -> dict[str, Any]:
        best_code = verify.pseudo_label or solution.outputs[0].extracted_answer
        user_msg = spec.prompt
        starter = spec.payload.get("starter_code")
        if starter:
            user_msg = f"{user_msg}\n\nStarter code:\n```python\n{starter}\n```"
        return {
            "task_id": spec.task_id,
            "domain": self.domain,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python engineer. Return clean, working Python code."
                    ),
                },
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"```python\n{best_code}\n```"},
            ],
        }


register_adapter("code", CodeUnitTestAdapter)
