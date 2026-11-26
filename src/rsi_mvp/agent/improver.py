"""Mutable meta-improver (v0): RSI recursive self-modification entry point."""

from __future__ import annotations

import json
from typing import Any

from rsi_mvp.agent.prompts import IMPROVER_SYSTEM_PROMPT
from rsi_mvp.agent.tools import parse_json_object


def _format_context_prompt(context: dict[str, Any]) -> str:
    parent_id = context.get("parent_id", "unknown")
    dev_summary = context.get("dev_summary", {})
    failed_cases = context.get("failed_cases", [])
    source_files: dict[str, str] = context.get("source_files", {})
    recent_history = context.get("recent_history", [])

    sections = [
        f"## Parent Candidate: {parent_id}",
        f"## Dev Summary\n```json\n{json.dumps(dev_summary, indent=2, ensure_ascii=False)}\n```",
    ]

    if recent_history:
        sections.append(
            "## Recent Iteration History (avoid repeating failed ideas)\n"
            f"```json\n{json.dumps(recent_history[-5:], indent=2, ensure_ascii=False)}\n```"
        )

    if failed_cases:
        sections.append(
            "## Sampled Failed Cases\n"
            f"```json\n{json.dumps(failed_cases, indent=2, ensure_ascii=False)}\n```"
        )

    code_blocks = ["## Current Agent Source Files (agent/)"]
    for rel_path in sorted(source_files):
        code_blocks.append(f"### {rel_path}\n```python\n{source_files[rel_path]}\n```")
    sections.append("\n\n".join(code_blocks))

    sections.append(
        "Propose ONE targeted, generalizable improvement as a strict JSON object "
        "matching the schema."
    )
    return "\n\n".join(sections)


def propose_improvement(context: dict[str, Any], llm_client: Any) -> dict[str, Any]:
    """Analyze parent traces + source code and return a ``MutationProposal`` dict."""
    user_prompt = _format_context_prompt(context)
    messages = [
        {"role": "system", "content": IMPROVER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = llm_client.chat(
        messages=messages,
        temperature=0.3,
        max_tokens=3072,
        response_format={"type": "json_object"},
    )
    proposal = parse_json_object(str(resp.content))
    for required_key in ("diagnosis", "hypothesis", "rationale", "edits"):
        if required_key not in proposal:
            raise ValueError(f"MutationProposal missing required field: {required_key!r}")
    return proposal
