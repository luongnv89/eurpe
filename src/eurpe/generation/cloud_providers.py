"""Cloud provider connection testing (issue #80).

Sends a minimal request to a cloud LLM provider to verify that the
supplied API key is valid and the selected model is accessible.

Supported providers
-------------------
* ``openai`` — ``POST https://api.openai.com/v1/chat/completions`` with
  ``max_tokens=1`` and a trivial prompt.
* ``anthropic`` — ``POST https://api.anthropic.com/v1/messages`` with
  ``max_tokens=1`` and a trivial prompt.
* ``gemini`` — ``POST https://generativelanguage.googleapis.com/v1beta/
  models/<model>:generateContent`` with a trivial prompt.
* ``openrouter`` — ``POST https://openrouter.ai/api/v1/chat/completions``
  (OpenAI-compatible) with ``max_tokens=1``.
* ``groq`` — ``POST https://api.groq.com/openai/v1/chat/completions``
  (OpenAI-compatible) with ``max_tokens=1``.

The test uses the smallest possible payload so token consumption is
negligible. A failure returns the provider's error detail so the UI can
surface it to the operator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"openai", "anthropic", "gemini", "openrouter", "groq"}

PROVIDER_ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
}


@dataclass
class ConnectionTestResult:
    """Outcome of a cloud provider connection test."""

    success: bool
    message: str
    model_confirmed: str | None = None
    error_detail: str | None = None


def _test_openai_compatible(
    provider: str,
    api_key: str,
    model: str,
    *,
    timeout: float = 15.0,
) -> ConnectionTestResult:
    """Test an OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq)."""
    url = PROVIDER_ENDPOINTS[provider]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/luongnv89/eurpe"
        headers["X-Title"] = "EURPE"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body, headers=headers)
    except httpx.ConnectError as exc:
        return ConnectionTestResult(
            success=False,
            message="Cannot reach the provider",
            error_detail=str(exc),
        )
    except httpx.ReadTimeout:
        return ConnectionTestResult(
            success=False,
            message="Request timed out",
            error_detail="The provider did not respond within the timeout period.",
        )
    except httpx.HTTPError as exc:
        return ConnectionTestResult(
            success=False,
            message="HTTP request failed",
            error_detail=str(exc),
        )

    if resp.status_code == 200:
        try:
            payload = resp.json()
            response_model = payload.get("model", model)
        except ValueError:
            response_model = model
        return ConnectionTestResult(
            success=True,
            message="Connection successful",
            model_confirmed=response_model,
        )

    try:
        detail = resp.json().get("error", {}).get("message", resp.text[:300])
    except ValueError:
        detail = resp.text[:300]
    return ConnectionTestResult(
        success=False,
        message=f"Provider returned HTTP {resp.status_code}",
        error_detail=detail,
    )


def _test_anthropic(api_key: str, model: str, *, timeout: float = 15.0) -> ConnectionTestResult:
    """Test the Anthropic Messages API."""
    url = PROVIDER_ENDPOINTS["anthropic"]
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body, headers=headers)
    except httpx.ConnectError as exc:
        return ConnectionTestResult(
            success=False,
            message="Cannot reach Anthropic",
            error_detail=str(exc),
        )
    except httpx.ReadTimeout:
        return ConnectionTestResult(
            success=False,
            message="Request timed out",
            error_detail="Anthropic did not respond within the timeout period.",
        )
    except httpx.HTTPError as exc:
        return ConnectionTestResult(
            success=False,
            message="HTTP request failed",
            error_detail=str(exc),
        )

    if resp.status_code == 200:
        return ConnectionTestResult(
            success=True,
            message="Connection successful",
            model_confirmed=model,
        )

    try:
        detail = resp.json().get("error", {}).get("message", resp.text[:300])
    except ValueError:
        detail = resp.text[:300]
    return ConnectionTestResult(
        success=False,
        message=f"Anthropic returned HTTP {resp.status_code}",
        error_detail=detail,
    )


def _test_gemini(api_key: str, model: str, *, timeout: float = 15.0) -> ConnectionTestResult:
    """Test the Google Gemini API."""
    url = PROVIDER_ENDPOINTS["gemini"].format(model=model)
    # Gemini appends the key as a query parameter
    params = {"key": api_key}
    body = {
        "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        "generationConfig": {"maxOutputTokens": 1},
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body, params=params)
    except httpx.ConnectError as exc:
        return ConnectionTestResult(
            success=False,
            message="Cannot reach Gemini",
            error_detail=str(exc),
        )
    except httpx.ReadTimeout:
        return ConnectionTestResult(
            success=False,
            message="Request timed out",
            error_detail="Gemini did not respond within the timeout period.",
        )
    except httpx.HTTPError as exc:
        return ConnectionTestResult(
            success=False,
            message="HTTP request failed",
            error_detail=str(exc),
        )

    if resp.status_code == 200:
        return ConnectionTestResult(
            success=True,
            message="Connection successful",
            model_confirmed=model,
        )

    try:
        detail = resp.json().get("error", {}).get("message", resp.text[:300])
    except ValueError:
        detail = resp.text[:300]
    return ConnectionTestResult(
        success=False,
        message=f"Gemini returned HTTP {resp.status_code}",
        error_detail=detail,
    )


def check_cloud_provider(
    provider: str,
    model: str,
    api_key: str,
    *,
    timeout: float = 15.0,
) -> ConnectionTestResult:
    """Test a cloud provider connection with the supplied credentials.

    Args:
        provider: One of ``openai``, ``anthropic``, ``gemini``,
            ``openrouter``, ``groq``.
        model: Model identifier (e.g. ``gpt-4o``, ``claude-sonnet-4-20250514``).
        api_key: The API key to validate.
        timeout: Seconds before the request is aborted.

    Returns:
        A :class:`ConnectionTestResult` with success/failure details.

    Raises:
        ValueError: If the provider is not in the supported set.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: {provider!r}. " f"Must be one of {sorted(SUPPORTED_PROVIDERS)}."
        )

    logger.info(
        "Testing cloud provider %s with model %s (key length=%d)",
        provider,
        model,
        len(api_key),
    )

    if provider == "anthropic":
        return _test_anthropic(api_key, model, timeout=timeout)
    if provider == "gemini":
        return _test_gemini(api_key, model, timeout=timeout)
    # OpenAI, OpenRouter, and Groq all use the OpenAI chat completions format
    return _test_openai_compatible(provider, api_key, model, timeout=timeout)
