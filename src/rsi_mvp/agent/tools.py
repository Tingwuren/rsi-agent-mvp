"""Self-contained utility tools used inside the mutable agent package."""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def strip_thinking_tags(text: str) -> str:
    """Remove optional ``<think>...</think>`` reasoning wrappers (e.g. Qwen3/DeepSeek)."""
    return _THINK_RE.sub("", text).strip()


def extract_code_block(raw_text: str) -> str:
    """Extract the last fenced Python code block or return cleaned raw text."""
    cleaned = strip_thinking_tags(raw_text)
    matches = _CODE_BLOCK_RE.findall(cleaned)
    if matches:
        return matches[-1].strip()
    return cleaned


def extract_boxed_answer(raw_text: str) -> str:
    """Extract the last balanced ``\\boxed{...}`` expression from a math response."""
    cleaned = strip_thinking_tags(raw_text)
    marker = "\\boxed{"
    boxed: list[str] = []
    idx = 0
    while True:
        pos = cleaned.find(marker, idx)
        if pos == -1:
            break
        start = pos + len(marker)
        depth = 1
        cur = start
        while cur < len(cleaned) and depth > 0:
            if cleaned[cur] == "{":
                depth += 1
            elif cleaned[cur] == "}":
                depth -= 1
            cur += 1
        if depth == 0:
            boxed.append(cleaned[start : cur - 1].strip())
            idx = cur
        else:
            break
    if boxed:
        return boxed[-1]
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def parse_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model output (handling markdown code fences)."""
    cleaned = strip_thinking_tags(raw_text)
    fenced = _JSON_BLOCK_RE.findall(cleaned)
    candidates = [fenced[-1].strip()] if fenced else []
    candidates.append(cleaned)

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])

    last_err: Exception | None = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception as exc:
            last_err = exc
    raise ValueError(f"Failed to parse JSON object from response: {last_err}")
