"""Tests for ``eurpe.generation.cloud_providers`` — cloud provider connection testing.

Tests use ``httpx.MockTransport`` so CI doesn't need real API keys or
network access. The production code sends real HTTP requests; the
mocked transport verifies the correct URL, headers, and body shape.
"""

from __future__ import annotations

import json

import httpx
import pytest

from eurpe.generation.cloud_providers import (
    SUPPORTED_PROVIDERS,
    check_cloud_provider,
)

# Save the real httpx.Client before any monkeypatching
_RealClient = httpx.Client


# ---------------------------------------------------------------------------
# Supported providers set
# ---------------------------------------------------------------------------


def test_supported_providers_contains_expected_keys() -> None:
    assert "openai" in SUPPORTED_PROVIDERS
    assert "anthropic" in SUPPORTED_PROVIDERS
    assert "gemini" in SUPPORTED_PROVIDERS
    assert "openrouter" in SUPPORTED_PROVIDERS
    assert "groq" in SUPPORTED_PROVIDERS


def test_unsupported_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        check_cloud_provider("bedrock", "anthropic.claude-v2", "sk-xxx")


# ---------------------------------------------------------------------------
# OpenAI — mocked transport
# ---------------------------------------------------------------------------


def _mock_openai_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["model"] == "gpt-4o"
    assert body["max_tokens"] == 1
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert request.headers["Authorization"].startswith("Bearer sk-test-")
    return httpx.Response(
        status_code=200,
        json={"model": "gpt-4o", "choices": [{"message": {"content": "Hi"}}]},
    )


def test_openai_success(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(_mock_openai_handler)

    class _MockClient:
        def __init__(self, **kwargs: object) -> None:
            self._client = _RealClient(transport=transport, **kwargs)

        def __enter__(self) -> _MockClient:
            self._client.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self._client.__exit__(*args)

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return self._client.post(*args, **kwargs)

    monkeypatch.setattr("eurpe.generation.cloud_providers.httpx.Client", _MockClient)
    result = check_cloud_provider("openai", "gpt-4o", "sk-test-key")
    assert result.success is True
    assert result.model_confirmed == "gpt-4o"


def _mock_openai_401_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        status_code=401,
        json={"error": {"message": "Invalid API key"}},
    )


def test_openai_invalid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(_mock_openai_401_handler)

    class _MockClient:
        def __init__(self, **kwargs: object) -> None:
            self._client = _RealClient(transport=transport, **kwargs)

        def __enter__(self) -> _MockClient:
            self._client.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self._client.__exit__(*args)

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return self._client.post(*args, **kwargs)

    monkeypatch.setattr("eurpe.generation.cloud_providers.httpx.Client", _MockClient)
    result = check_cloud_provider("openai", "gpt-4o", "sk-bad-key")
    assert result.success is False
    assert "401" in result.message
    assert "Invalid API key" in result.error_detail


# ---------------------------------------------------------------------------
# Anthropic — mocked transport
# ---------------------------------------------------------------------------


def _mock_anthropic_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["model"] == "claude-sonnet-4-20250514"
    assert body["max_tokens"] == 1
    assert request.headers["x-api-key"] == "sk-ant-test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    return httpx.Response(
        status_code=200,
        json={"content": [{"text": "Hi"}]},
    )


def test_anthropic_success(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(_mock_anthropic_handler)

    class _MockClient:
        def __init__(self, **kwargs: object) -> None:
            self._client = _RealClient(transport=transport, **kwargs)

        def __enter__(self) -> _MockClient:
            self._client.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self._client.__exit__(*args)

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return self._client.post(*args, **kwargs)

    monkeypatch.setattr("eurpe.generation.cloud_providers.httpx.Client", _MockClient)
    result = check_cloud_provider("anthropic", "claude-sonnet-4-20250514", "sk-ant-test")
    assert result.success is True


def _mock_anthropic_403_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        status_code=403,
        json={"error": {"message": "Invalid API key provided"}},
    )


def test_anthropic_invalid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(_mock_anthropic_403_handler)

    class _MockClient:
        def __init__(self, **kwargs: object) -> None:
            self._client = _RealClient(transport=transport, **kwargs)

        def __enter__(self) -> _MockClient:
            self._client.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self._client.__exit__(*args)

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return self._client.post(*args, **kwargs)

    monkeypatch.setattr("eurpe.generation.cloud_providers.httpx.Client", _MockClient)
    result = check_cloud_provider("anthropic", "claude-sonnet-4-20250514", "sk-ant-bad")
    assert result.success is False
    assert "403" in result.message


# ---------------------------------------------------------------------------
# Gemini — mocked transport
# ---------------------------------------------------------------------------


def _mock_gemini_handler(request: httpx.Request) -> httpx.Response:
    assert "gemini-2.5-flash" in str(request.url)
    assert "key=test-gemini-key" in str(request.url)
    body = json.loads(request.content)
    assert body["generationConfig"]["maxOutputTokens"] == 1
    return httpx.Response(
        status_code=200,
        json={"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]},
    )


def test_gemini_success(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(_mock_gemini_handler)

    class _MockClient:
        def __init__(self, **kwargs: object) -> None:
            self._client = _RealClient(transport=transport, **kwargs)

        def __enter__(self) -> _MockClient:
            self._client.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self._client.__exit__(*args)

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return self._client.post(*args, **kwargs)

    monkeypatch.setattr("eurpe.generation.cloud_providers.httpx.Client", _MockClient)
    result = check_cloud_provider("gemini", "gemini-2.5-flash", "test-gemini-key")
    assert result.success is True
    assert result.model_confirmed == "gemini-2.5-flash"


def _mock_gemini_400_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        status_code=400,
        json={"error": {"message": "API key not valid"}},
    )


def test_gemini_invalid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(_mock_gemini_400_handler)

    class _MockClient:
        def __init__(self, **kwargs: object) -> None:
            self._client = _RealClient(transport=transport, **kwargs)

        def __enter__(self) -> _MockClient:
            self._client.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self._client.__exit__(*args)

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return self._client.post(*args, **kwargs)

    monkeypatch.setattr("eurpe.generation.cloud_providers.httpx.Client", _MockClient)
    result = check_cloud_provider("gemini", "gemini-2.5-flash", "bad-key")
    assert result.success is False
    assert "400" in result.message


# ---------------------------------------------------------------------------
# Groq — OpenAI-compatible
# ---------------------------------------------------------------------------


def _mock_groq_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["model"] == "llama-3.3-70b-versatile"
    assert request.headers["Authorization"].startswith("Bearer gsk_")
    return httpx.Response(
        status_code=200,
        json={"model": "llama-3.3-70b-versatile", "choices": [{"message": {"content": "Hi"}}]},
    )


def test_groq_success(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(_mock_groq_handler)

    class _MockClient:
        def __init__(self, **kwargs: object) -> None:
            self._client = _RealClient(transport=transport, **kwargs)

        def __enter__(self) -> _MockClient:
            self._client.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self._client.__exit__(*args)

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return self._client.post(*args, **kwargs)

    monkeypatch.setattr("eurpe.generation.cloud_providers.httpx.Client", _MockClient)
    result = check_cloud_provider("groq", "llama-3.3-70b-versatile", "gsk_test")
    assert result.success is True


# ---------------------------------------------------------------------------
# OpenRouter — OpenAI-compatible with extra headers
# ---------------------------------------------------------------------------


def _mock_openrouter_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-4o"
    assert request.headers["Authorization"].startswith("Bearer sk-or-")
    assert request.headers["HTTP-Referer"] == "https://github.com/luongnv89/eurpe"
    assert request.headers["X-Title"] == "EURPE"
    return httpx.Response(
        status_code=200,
        json={"model": "openai/gpt-4o", "choices": [{"message": {"content": "Hi"}}]},
    )


def test_openrouter_success(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(_mock_openrouter_handler)

    class _MockClient:
        def __init__(self, **kwargs: object) -> None:
            self._client = _RealClient(transport=transport, **kwargs)

        def __enter__(self) -> _MockClient:
            self._client.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self._client.__exit__(*args)

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return self._client.post(*args, **kwargs)

    monkeypatch.setattr("eurpe.generation.cloud_providers.httpx.Client", _MockClient)
    result = check_cloud_provider("openrouter", "openai/gpt-4o", "sk-or-v1-test")
    assert result.success is True


# ---------------------------------------------------------------------------
# Connection error handling
# ---------------------------------------------------------------------------


def test_connect_error_returns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BoomClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _BoomClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "eurpe.generation.cloud_providers.httpx.Client",
        _BoomClient,
    )
    result = check_cloud_provider("openai", "gpt-4o", "sk-test")
    assert result.success is False
    assert "Cannot reach" in result.message
