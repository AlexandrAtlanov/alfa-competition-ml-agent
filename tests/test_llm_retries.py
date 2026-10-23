"""Ретраи LLM на 429: локальный Groq ждёт 30 с и делает 3 повтора."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from agent.llm import LLMConfig, _chat_with_retries


class _RateLimitedCompletions:
    """Первые `failures` вызовов падают с 429, затем отдают успешный ответ."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            response = httpx.Response(429, request=request)
            raise openai.RateLimitError("rate limited", response=response, body=None)
        return "response"


class _Log:
    def log(self, **kwargs):
        pass


def _config() -> LLMConfig:
    return LLMConfig(
        api_key="test",
        base_url="https://api.groq.com/openai/v1",
        model="test-model",
        timeout=10,
        code_timeout=10,
        max_retries=5,
        rate_limit_retries=3,
        rate_limit_wait=30,
        temperature=0.2,
        max_tokens=None,
    )


def test_rate_limit_retries_with_fixed_30s_wait(monkeypatch):
    completions = _RateLimitedCompletions(failures=3)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sleeps: list[float] = []
    monkeypatch.setattr("agent.llm.time.sleep", sleeps.append)

    response, _ = _chat_with_retries(client, {}, _config(), _Log())

    assert response == "response"
    assert completions.calls == 4
    assert sleeps == [30.0, 30.0, 30.0]


def test_rate_limit_gives_up_after_three_retries(monkeypatch):
    completions = _RateLimitedCompletions(failures=99)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sleeps: list[float] = []
    monkeypatch.setattr("agent.llm.time.sleep", sleeps.append)

    with pytest.raises(openai.RateLimitError):
        _chat_with_retries(client, {}, _config(), _Log())

    assert completions.calls == 4
    assert sleeps == [30.0, 30.0, 30.0]
