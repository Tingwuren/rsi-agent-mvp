"""Unit tests for LLM clients and token budget enforcement."""

from __future__ import annotations

import pytest

from rsi_mvp.harness.llm import BudgetExceededError, MockLLMClient


def test_mock_llm_client_returns_scripted_responses_and_tracks_tokens() -> None:
    client = MockLLMClient(responses=["first", "second"], tokens_per_call=(10, 5))
    r1 = client.chat([{"role": "user", "content": "hi"}])
    r2 = client.chat([{"role": "user", "content": "again"}])
    assert r1.content == "first"
    assert r2.content == "second"
    assert client.total_tokens_used == 30
    assert len(client.call_history) == 2

    client.reset_usage()
    assert client.total_tokens_used == 0
    assert client.call_history == []


def test_mock_llm_client_enforces_token_budget() -> None:
    client = MockLLMClient(responses=["ok"], max_total_tokens=20, tokens_per_call=(15, 10))
    with pytest.raises(BudgetExceededError):
        client.chat([{"role": "user", "content": "over budget"}])
