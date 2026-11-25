"""Static architectural boundary & security guard tests."""

from __future__ import annotations

import ast
from pathlib import Path


def test_mutable_agent_never_imports_harness() -> None:
    """Enforce strict trust boundary: ``agent/`` must never import ``rsi_mvp.harness``."""
    agent_dir = Path("src/rsi_mvp/agent")
    assert agent_dir.is_dir()

    violations: list[str] = []
    for py_file in agent_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "harness" in alias.name.split("."):
                        violations.append(f"{py_file}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "harness" in module.split("."):
                    violations.append(f"{py_file}: from {module} import ...")

    assert not violations, f"Trust boundary violated: {violations}"
