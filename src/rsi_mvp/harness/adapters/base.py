"""TaskAdapter protocol and plugin registry."""

from __future__ import annotations

from typing import Any, Protocol

from rsi_mvp.schemas import TaskInput, TaskSolution, TaskSpec, VerifyResult


class TaskAdapter(Protocol):
    """Adapts a domain-specific task to the generic harness pipeline."""

    domain: str

    def to_task_input(self, spec: TaskSpec, num_samples: int = 1) -> TaskInput: ...

    def extract_answer(self, raw_text: str, spec: TaskSpec) -> str: ...

    def format_for_sft(
        self, spec: TaskSpec, solution: TaskSolution, verify: VerifyResult
    ) -> dict[str, Any]: ...


_ADAPTER_REGISTRY: dict[str, type[Any]] = {}


def register_adapter(name: str, adapter_cls: type[Any]) -> None:
    _ADAPTER_REGISTRY[name] = adapter_cls


def get_adapter(name: str) -> TaskAdapter:
    if name not in _ADAPTER_REGISTRY:
        available = sorted(_ADAPTER_REGISTRY)
        raise KeyError(f"Unknown TaskAdapter {name!r}. Available: {available}")
    return _ADAPTER_REGISTRY[name]()


def list_adapters() -> list[str]:
    return sorted(_ADAPTER_REGISTRY)
