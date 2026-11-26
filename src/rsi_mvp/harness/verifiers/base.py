"""Verifier protocol and plugin registry."""

from __future__ import annotations

from typing import Any, Protocol

from rsi_mvp.schemas import TaskSolution, TaskSpec, VerifyResult


class Verifier(Protocol):
    """Evaluates a ``TaskSolution`` against a ``TaskSpec`` (labeled or label-free)."""

    mode_name: str
    is_label_free: bool

    def verify(self, spec: TaskSpec, solution: TaskSolution) -> VerifyResult: ...


_VERIFIER_REGISTRY: dict[str, type[Any]] = {}


def register_verifier(name: str, verifier_cls: type[Any]) -> None:
    _VERIFIER_REGISTRY[name] = verifier_cls


def get_verifier(name: str) -> Verifier:
    if name not in _VERIFIER_REGISTRY:
        available = sorted(_VERIFIER_REGISTRY)
        raise KeyError(f"Unknown Verifier {name!r}. Available: {available}")
    return _VERIFIER_REGISTRY[name]()


def list_verifiers() -> list[str]:
    return sorted(_VERIFIER_REGISTRY)
