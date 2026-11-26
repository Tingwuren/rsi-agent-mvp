"""LLM client abstractions: OpenAI-compatible client, Mock client, and budget guards."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from openai import OpenAI

from rsi_mvp.schemas import LLMResponse


class BudgetExceededError(RuntimeError):
    """Raised when cumulative LLM token consumption exceeds the configured hard cap."""


class BaseLLMClient(Protocol):
    """Protocol implemented by all production and mock LLM clients."""

    @property
    def total_tokens_used(self) -> int: ...

    def reset_usage(self) -> None: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


class OpenAICompatibleClient:
    """OpenAI-compatible chat client supporting local vLLM (with LoRA) and cloud APIs."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_total_tokens: int | None = None,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.5,
        timeout_sec: float = 120.0,
    ) -> None:
        self.model = model
        resolved_base_url = base_url or os.environ.get(
            "RSI_LLM_BASE_URL", "http://127.0.0.1:8000/v1"
        )
        resolved_api_key = api_key or os.environ.get("RSI_LLM_API_KEY", "EMPTY")
        self.base_url = resolved_base_url
        self.max_total_tokens = max_total_tokens
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self._client = OpenAI(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            timeout=timeout_sec,
        )
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def total_tokens_used(self) -> int:
        with self._lock:
            return self._prompt_tokens + self._completion_tokens

    def reset_usage(self) -> None:
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._prompt_tokens += max(0, prompt_tokens)
            self._completion_tokens += max(0, completion_tokens)
            total = self._prompt_tokens + self._completion_tokens
        if self.max_total_tokens is not None and total > self.max_total_tokens:
            raise BudgetExceededError(
                f"LLM token budget exceeded: {total} > {self.max_total_tokens}"
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self.max_total_tokens is not None and self.total_tokens_used >= self.max_total_tokens:
            raise BudgetExceededError(
                f"LLM token budget already exhausted: "
                f"{self.total_tokens_used} >= {self.max_total_tokens}"
            )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(**kwargs)
                latency_ms = int((time.perf_counter() - start) * 1000)
                content = completion.choices[0].message.content or ""
                usage = completion.usage
                p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                self._record_usage(p_tok, c_tok)
                return LLMResponse(
                    content=content,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    latency_ms=latency_ms,
                    model=getattr(completion, "model", self.model) or self.model,
                )
            except BudgetExceededError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_sec * (2**attempt))

        raise RuntimeError(
            f"OpenAICompatibleClient failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc


class MockLLMClient:
    """Deterministic in-memory LLM client for fast offline tests and smoke checks."""

    def __init__(
        self,
        responses: list[str] | None = None,
        handler: Callable[[list[dict[str, str]], float, int], str] | None = None,
        max_total_tokens: int | None = None,
        tokens_per_call: tuple[int, int] = (50, 30),
        model_name: str = "mock-llm",
    ) -> None:
        self._responses = list(responses or [])
        self._handler = handler
        self._cursor = 0
        self.max_total_tokens = max_total_tokens
        self._tokens_per_call = tokens_per_call
        self.model_name = model_name
        self._lock = threading.Lock()
        self._total_tokens = 0
        self.call_history: list[dict[str, Any]] = []

    @property
    def total_tokens_used(self) -> int:
        with self._lock:
            return self._total_tokens

    def reset_usage(self) -> None:
        with self._lock:
            self._total_tokens = 0
            self._cursor = 0
            self.call_history.clear()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        with self._lock:
            if self.max_total_tokens is not None and self._total_tokens >= self.max_total_tokens:
                raise BudgetExceededError(
                    f"Mock token budget exhausted: {self._total_tokens} >= {self.max_total_tokens}"
                )
            self.call_history.append(
                {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": response_format,
                }
            )
            if self._handler is not None:
                content = self._handler(messages, temperature, max_tokens)
            elif self._responses:
                idx = min(self._cursor, len(self._responses) - 1)
                content = self._responses[idx]
                self._cursor += 1
            else:
                content = ""

            p_tok, c_tok = self._tokens_per_call
            self._total_tokens += p_tok + c_tok
            if self.max_total_tokens is not None and self._total_tokens > self.max_total_tokens:
                raise BudgetExceededError(
                    f"Mock token budget exceeded: {self._total_tokens} > {self.max_total_tokens}"
                )

        return LLMResponse(
            content=content,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            latency_ms=1,
            model=self.model_name,
        )
