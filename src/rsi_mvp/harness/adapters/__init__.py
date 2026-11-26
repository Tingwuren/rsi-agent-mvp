"""TaskAdapter plugins (code, math, qa)."""

from rsi_mvp.harness.adapters.base import TaskAdapter, get_adapter, list_adapters, register_adapter
from rsi_mvp.harness.adapters.code import CodeUnitTestAdapter, extract_python_code
from rsi_mvp.harness.adapters.math import MathSymbolicAdapter, extract_boxed_answer
from rsi_mvp.harness.adapters.qa import GenericQAAdapter

__all__ = [
    "CodeUnitTestAdapter",
    "GenericQAAdapter",
    "MathSymbolicAdapter",
    "TaskAdapter",
    "extract_boxed_answer",
    "extract_python_code",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
