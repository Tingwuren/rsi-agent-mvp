"""Mutable seed solver (v0): domain-aware single/multi-sample solver."""

from __future__ import annotations

import time
from typing import Any

from rsi_mvp.agent.prompts import SOLVER_SYSTEM_PROMPTS
from rsi_mvp.agent.tools import extract_boxed_answer, extract_code_block, strip_thinking_tags


def _build_user_message(task: dict[str, Any]) -> str:
    prompt = str(task.get("prompt", ""))
    payload = task.get("payload") or {}
    starter = payload.get("starter_code")
    if starter:
        return f"{prompt}\n\nStarter code:\n```python\n{starter}\n```"
    return prompt


def _extract_by_domain(domain: str, raw_text: str) -> str:
    if domain == "code":
        return extract_code_block(raw_text)
    if domain == "math":
        return extract_boxed_answer(raw_text)
    return strip_thinking_tags(raw_text)


def solve(task: dict[str, Any], llm_client: Any) -> dict[str, Any]:
    """Solve a sanitized ``TaskInput`` dict and return a ``TaskSolution`` dict."""
    start = time.perf_counter()
    task_id = str(task["task_id"])
    domain = str(task.get("domain", "qa"))
    num_samples = max(1, int(task.get("num_samples", 1)))

    system_prompt = SOLVER_SYSTEM_PROMPTS.get(domain, SOLVER_SYSTEM_PROMPTS["qa"])
    user_message = _build_user_message(task)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    outputs: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    total_tokens = 0
    # Use higher temperature automatically when drawing multiple samples for consensus
    temperature = 0.2 if num_samples == 1 else 0.7

    for idx in range(num_samples):
        resp = llm_client.chat(messages=messages, temperature=temperature, max_tokens=4096)
        raw_text = str(resp.content)
        extracted = _extract_by_domain(domain, raw_text)
        p_tok = int(getattr(resp, "prompt_tokens", 0))
        c_tok = int(getattr(resp, "completion_tokens", 0))
        lat_ms = int(getattr(resp, "latency_ms", 0))
        total_tokens += p_tok + c_tok

        outputs.append(
            {
                "raw_text": raw_text,
                "extracted_answer": extracted,
                "auxiliary": {},
            }
        )
        steps.append(
            {
                "step_index": idx,
                "action_type": "llm_call",
                "input_summary": user_message[:300],
                "output_summary": extracted[:300],
                "tool_name": None,
                "latency_ms": lat_ms,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
            }
        )

    return {
        "task_id": task_id,
        "outputs": outputs,
        "steps": steps,
        "error": None,
        "total_tokens": total_tokens,
        "elapsed_sec": round(time.perf_counter() - start, 4),
    }
