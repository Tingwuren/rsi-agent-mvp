"""Mutable seed agent template (v0).

Strict architectural boundary: modules in this package must NEVER import
from ``rsi_mvp.harness``.
"""

from rsi_mvp.agent.core import solve
from rsi_mvp.agent.improver import propose_improvement

__all__ = ["propose_improvement", "solve"]
