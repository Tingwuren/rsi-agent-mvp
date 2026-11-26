#!/usr/bin/env python3
"""Read-only worker entrypoint executed inside SubprocessSandbox or DockerSandbox.

Deliberately depends ONLY on the Python standard library so it works inside
minimal container images (e.g. ``python:3.11-slim``) and never exposes
``rsi_mvp.harness`` to untrusted mutated agent code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class _WorkerLLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    model: str = ""


class _WorkerMockLLMClient:
    """Self-contained mock LLM client for sandboxed smoke tests and offline runs."""

    def __init__(self, responses: list[str], model: str = "mock-llm") -> None:
        self._responses = list(responses)
        self._cursor = 0
        self._model = model
        self._total_tokens = 0

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens

    def reset_usage(self) -> None:
        self._total_tokens = 0
        self._cursor = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> _WorkerLLMResponse:
        _ = (messages, temperature, max_tokens, response_format)
        if not self._responses:
            content = ""
        else:
            idx = min(self._cursor, len(self._responses) - 1)
            content = self._responses[idx]
            self._cursor += 1
        self._total_tokens += 40
        return _WorkerLLMResponse(
            content=content,
            prompt_tokens=25,
            completion_tokens=15,
            latency_ms=1,
            model=self._model,
        )


class _WorkerOpenAIHTTPClient:
    """Minimal stdlib OpenAI-compatible HTTP client for sandboxed execution."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: float = 120.0,
    ) -> None:
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._timeout_sec = timeout_sec
        self._total_tokens = 0

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens

    def reset_usage(self) -> None:
        self._total_tokens = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> _WorkerLLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format

        req = urllib.request.Request(
            self._endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency_ms = int((time.perf_counter() - start) * 1000)

        content = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens", 0) or 0)
        c_tok = int(usage.get("completion_tokens", 0) or 0)
        self._total_tokens += p_tok + c_tok
        return _WorkerLLMResponse(
            content=content,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            latency_ms=latency_ms,
            model=str(data.get("model", self._model)),
        )


def _load_isolated_agent_package(snapshot_dir: Path) -> types.ModuleType:
    """Mount ``<snapshot_dir>/agent`` as ``rsi_mvp.agent`` without exposing harness."""
    agent_dir = (snapshot_dir / "agent").resolve()
    if not (agent_dir / "__init__.py").is_file():
        raise FileNotFoundError(f"Snapshot missing agent/__init__.py: {agent_dir}")

    # Purge any previously loaded rsi_mvp modules so mutations are always loaded fresh
    for mod_name in [m for m in list(sys.modules) if m == "rsi_mvp" or m.startswith("rsi_mvp.")]:
        del sys.modules[mod_name]

    # Register synthetic namespace package pointing ONLY to snapshot_dir
    pkg = types.ModuleType("rsi_mvp")
    pkg.__path__ = [str(snapshot_dir.resolve())]
    sys.modules["rsi_mvp"] = pkg

    return importlib.import_module("rsi_mvp.agent")


def _build_client(llm_spec: dict[str, Any]) -> Any:
    mode = llm_spec.get("mode", "mock")
    if mode == "mock":
        return _WorkerMockLLMClient(
            responses=list(llm_spec.get("mock_responses") or []),
            model=str(llm_spec.get("model", "mock-llm")),
        )
    if mode == "openai_http":
        return _WorkerOpenAIHTTPClient(
            base_url=str(llm_spec["base_url"]),
            api_key=str(llm_spec.get("api_key", "EMPTY")),
            model=str(llm_spec["model"]),
            timeout_sec=float(llm_spec.get("timeout_sec", 120.0)),
        )
    raise ValueError(f"Unsupported worker LLM mode: {mode!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated RSI agent worker")
    parser.add_argument("--mode", required=True, choices=["solve", "improve"])
    parser.add_argument("--snapshot", required=True, help="Path containing agent/ directory")
    parser.add_argument("--input", required=True, help="JSON input payload path")
    parser.add_argument("--output", required=True, help="JSON output payload path")
    args = parser.parse_args()

    out_path = Path(args.output)
    try:
        raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
        llm_client = _build_client(raw["llm_spec"])
        agent_mod = _load_isolated_agent_package(Path(args.snapshot))

        if args.mode == "solve":
            if not hasattr(agent_mod, "solve"):
                raise AttributeError("Snapshot agent missing required 'solve' entry point")
            result = agent_mod.solve(raw["payload"], llm_client)
        else:
            if not hasattr(agent_mod, "propose_improvement"):
                raise AttributeError(
                    "Snapshot agent missing required 'propose_improvement' entry point"
                )
            result = agent_mod.propose_improvement(raw["payload"], llm_client)

        if not isinstance(result, dict):
            raise TypeError(f"Agent {args.mode}() must return dict, got {type(result).__name__}")

        out_path.write_text(
            json.dumps({"ok": True, "result": result}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        out_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
