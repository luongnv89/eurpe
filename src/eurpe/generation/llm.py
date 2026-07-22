"""LLM client backends for the generation layer.

Two concrete implementations sit behind the :class:`LLMClient` Protocol:

* :class:`OllamaLLMClient` — calls a local Ollama daemon at
  ``POST /api/generate``. The daemon listens on ``localhost`` so this
  is "local" traffic; ``offline_mode`` does NOT block it. Wraps
  connection failure in :class:`LLMUnavailableError` with an
  actionable message ("start ``ollama serve`` and pull the model with
  ``ollama pull <model>``").
* :class:`DeterministicLLMClient` — pure-Python, zero-network. Reflects
  the prompt's structured ``[N]`` citation markers back into the
  output so the citation-rendering code path is exercised. Used by
  the test suite AND as the offline fallback when Ollama is
  unreachable in offline mode (analogous to
  :class:`~eurpe.retrieval.embeddings.DeterministicHashEmbedder`).

The :func:`make_llm_client` factory mirrors
:func:`eurpe.retrieval.embeddings.make_embedder`: in offline mode it
probes the Ollama TCP port and falls back to the deterministic stub
if the daemon is not running. In non-offline mode the factory trusts
the caller and returns the real client; surfacing connection errors
at the first ``generate()`` call gives a clearer trace than failing
at construction.

Why the deterministic stub is appropriate for tests
---------------------------------------------------
The same reasoning as the deterministic-hash embedder: real LLMs are
slow, depend on a model download, and produce non-reproducible
output. The acceptance criterion "Generation can run using a local
model runtime without cloud API calls" reduces, in CI, to "the
workflow runs end-to-end without socket connections". The
deterministic stub gives us a stable, side-effect-free stand-in we
can run under the ``no_network`` fixture. Its drafting *quality* is
intentionally poor — it merely echoes the prompt — but that is
enough to verify the wiring.
"""

from __future__ import annotations

import logging
import os
import re
import socket
from typing import Protocol
from urllib.parse import urlparse

import httpx

from eurpe.generation.errors import GenerationError, LLMUnavailableError
from eurpe.security import (
    EgressDeniedError,
    NetworkPolicyGate,
    make_network_policy,
)
from eurpe.security.policy import _is_loopback

logger = logging.getLogger(__name__)


_OPENAI_COMPATIBLE_DEFAULTS: dict[str, tuple[str, str | None]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "lmstudio": ("http://localhost:1234/v1", None),
    "vllm": ("http://localhost:8000/v1", "VLLM_API_KEY"),
    "llamacpp": ("http://localhost:8080/v1", None),
}

_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
_GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"


def _url_port(parsed: object) -> int:
    """Return the effective TCP port for a parsed HTTP(S) URL."""

    port = getattr(parsed, "port", None)
    if port is not None:
        return int(port)
    scheme = (getattr(parsed, "scheme", "") or "http").lower()
    return 443 if scheme == "https" else 80


def _requires_policy(base_url: str) -> bool:
    """Return whether ``base_url`` needs an explicit network gate."""

    parsed = urlparse(base_url)
    return not _is_loopback(parsed.hostname or "")


def _check_policy_or_raise(
    *,
    policy: NetworkPolicyGate | None,
    base_url: str,
    path: str,
    source: str,
    provider: str,
    default_scheme: str,
) -> None:
    """Gate non-loopback HTTP before any prompt body reaches the transport."""

    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    if policy is None:
        if _is_loopback(host):
            return
        raise LLMUnavailableError(
            f"{provider} requires a NetworkPolicyGate before contacting "
            f"{host}:{_url_port(parsed)}. Use make_llm_client(config) so "
            "non-loopback LLM traffic is checked against network_allowlist."
        )
    policy.check(
        host=host,
        port=_url_port(parsed),
        scheme=parsed.scheme or default_scheme,
        path=path,
        source=source,
    )


def _read_api_key(*, provider: str, env_var: str | None, required: bool) -> str | None:
    """Read a provider secret from the environment without logging its value."""

    if not env_var:
        return None
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    if required:
        raise LLMUnavailableError(
            f"{provider} requires an API key. Set the {env_var} environment variable "
            "and retry. API keys must not be stored in config.yaml."
        )
    return None


class LLMClient(Protocol):
    """Synchronous text-generation interface.

    Concrete implementations MUST be safe to call from a single
    thread; the workflow generates one section at a time.
    """

    @property
    def model(self) -> str:
        """Identifier of the underlying model.

        Recorded in :class:`~eurpe.generation.GenerationDraft.model`
        so a stored draft is traceable to the runtime that produced
        it.
        """

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """Return the model's text completion of ``prompt``."""


class _PooledHTTPClientMixin:
    """One reusable ``httpx.Client`` per LLM client instance.

    A fresh ``httpx.Client`` per ``generate()`` call means a fresh TCP
    (and, for cloud providers, TLS) handshake per request — the critic
    loop makes up to 10 LLM calls per draft, so connection reuse
    matters. Created lazily so tests that monkeypatch ``httpx.Client``
    after constructing the LLM client still get the fake.
    """

    _timeout: float
    _pooled_client: httpx.Client | None = None

    def _http_client(self) -> httpx.Client:
        if self._pooled_client is None:
            self._pooled_client = httpx.Client(timeout=self._timeout)
        return self._pooled_client

    def close(self) -> None:
        """Release the pooled connection; safe to call more than once."""

        if self._pooled_client is not None:
            self._pooled_client.close()
            self._pooled_client = None


# ---------------------------------------------------------------------------
# Ollama client (real model, talks to localhost)
# ---------------------------------------------------------------------------


class OllamaLLMClient(_PooledHTTPClientMixin):
    """LLM client backed by a local Ollama daemon (``POST /api/generate``).

    Talks to ``localhost:11434`` by default — what a developer machine
    running ``ollama serve`` exposes. ``offline_mode`` does NOT block
    this; the offline contract is "no outbound *Internet* traffic".
    The :func:`make_llm_client` factory falls back to
    :class:`DeterministicLLMClient` when Ollama is unreachable in
    offline mode, so callers in stricter environments still get a
    working pipeline.

    Why ``"stream": false`` is mandatory
    ------------------------------------
    The ``/api/generate`` endpoint defaults to streaming chunked
    JSON. That breaks ``response.json()`` because the body is a
    sequence of JSON objects rather than a single document. Setting
    ``stream: false`` returns one consolidated JSON object whose
    ``response`` field carries the full completion. This is the
    single most common cause of "Ollama returned a malformed
    response" errors.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        policy: NetworkPolicyGate | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        # Optional, default-None for backward compatibility with the
        # large set of existing tests that build OllamaLLMClient
        # directly. The factory always wires the gate; only
        # production paths consult it.
        self._policy = policy

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        """Read-only access to the configured Ollama URL — useful for tests / logs."""

        return self._base_url

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """Call Ollama's ``/api/generate`` and return the completion text.

        Raises :class:`LLMUnavailableError` on connection / timeout
        failures (the daemon isn't running or is overloaded) and
        :class:`GenerationError` on protocol / status errors (malformed
        JSON, 4xx / 5xx response). Both are subclasses of
        :class:`GenerationError`, so a caller that only cares about
        "the generation layer broke" can catch the base.
        """

        url = f"{self._base_url}/api/generate"
        body = {
            "model": self._model,
            "prompt": prompt,
            # ``stream: false`` is non-negotiable — see class docstring.
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        # Gate FIRST so a deny never reaches httpx with the prompt
        # payload. The gate's audit log records the attempt but never
        # the prompt body — that contract is enforced inside
        # eurpe.security.audit, not here.
        if self._policy is not None:
            parsed = urlparse(self._base_url)
            self._policy.check(
                host=parsed.hostname or "localhost",
                port=parsed.port or 11434,
                scheme=parsed.scheme or "http",
                path="/api/generate",
                source="ollama_llm.generate",
            )
        try:
            resp = self._http_client().post(url, json=body)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            # Connection-level failures are recoverable by the user
            # ("start ollama serve") — surface a distinct error type
            # so the CLI can print an actionable message.
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self._base_url}: {exc}. "
                f"Start the daemon with `ollama serve` and ensure the "
                f"model is pulled (`ollama pull {self._model}`)."
            ) from exc
        except httpx.HTTPError as exc:
            # Other transport problems (proxy, malformed URL) — protocol-
            # level rather than reachability — get the generic error.
            raise GenerationError(f"Ollama request to {url} failed: {exc}") from exc

        if resp.status_code >= 400:
            # Truncate the body in the error message so a giant 500
            # response doesn't drown the log. 500 chars is enough to see
            # the key / value pair Ollama complained about.
            body_preview = resp.text[:500]
            raise GenerationError(
                f"Ollama returned HTTP {resp.status_code} for model {self._model!r}: {body_preview}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise GenerationError(
                f"Ollama returned non-JSON body for model {self._model!r}: {resp.text[:500]}"
            ) from exc

        completion = payload.get("response")
        if not isinstance(completion, str) or not completion:
            raise GenerationError(
                f"Ollama returned a malformed payload for model {self._model!r}: {payload!r}"
            )
        return completion


# ---------------------------------------------------------------------------
# OpenAI-compatible chat-completions clients
# ---------------------------------------------------------------------------


class OpenAICompatibleLLMClient(_PooledHTTPClientMixin):
    """LLM client for providers exposing ``/chat/completions``.

    Covers OpenAI, OpenRouter, Groq, LM Studio, vLLM, and llama.cpp.
    The client uses raw ``httpx`` calls instead of provider SDKs so the
    dependency graph stays small and every request can pass through
    :class:`NetworkPolicyGate` before prompt content reaches the
    transport layer.
    """

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        policy: NetworkPolicyGate | None = None,
    ) -> None:
        self._provider = provider
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._policy = policy

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def base_url(self) -> str:
        return self._base_url

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        path = "/chat/completions"
        url = f"{self._base_url}{path}"
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        _check_policy_or_raise(
            policy=self._policy,
            base_url=self._base_url,
            path=path,
            source=f"{self._provider}_llm.generate",
            provider=self._provider,
            default_scheme="http",
        )

        try:
            resp = self._http_client().post(url, json=body, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            raise LLMUnavailableError(
                f"Cannot reach {self._provider} LLM endpoint at {self._base_url}: {exc}."
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationError(
                f"{self._provider} request to {self._base_url} failed: {exc}"
            ) from exc

        if resp.status_code >= 400:
            raise GenerationError(
                f"{self._provider} returned HTTP {resp.status_code} for "
                f"model {self._model!r}: {resp.text[:500]}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise GenerationError(
                f"{self._provider} returned non-JSON body for model "
                f"{self._model!r}: {resp.text[:500]}"
            ) from exc

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GenerationError(
                f"{self._provider} returned a malformed payload for "
                f"model {self._model!r}: {payload!r}"
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise GenerationError(
                f"{self._provider} returned a malformed choice for model {self._model!r}."
            )
        message = first.get("message")
        completion: object
        if isinstance(message, dict):
            completion = message.get("content")
        else:
            completion = first.get("text")
        if not isinstance(completion, str) or not completion:
            raise GenerationError(
                f"{self._provider} returned an empty completion for model {self._model!r}."
            )
        return completion


class AnthropicLLMClient(_PooledHTTPClientMixin):
    """LLM client for Anthropic's Messages API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 120.0,
        policy: NetworkPolicyGate | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._policy = policy

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        path = "/v1/messages"
        url = f"{self._base_url}{path}"
        body = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        _check_policy_or_raise(
            policy=self._policy,
            base_url=self._base_url,
            path=path,
            source="anthropic_llm.generate",
            provider="Anthropic",
            default_scheme="https",
        )

        try:
            resp = self._http_client().post(url, json=body, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            raise LLMUnavailableError(
                f"Cannot reach Anthropic LLM endpoint at {self._base_url}: {exc}."
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationError(f"Anthropic request to {self._base_url} failed: {exc}") from exc

        if resp.status_code >= 400:
            raise GenerationError(
                f"Anthropic returned HTTP {resp.status_code} for "
                f"model {self._model!r}: {resp.text[:500]}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise GenerationError(
                f"Anthropic returned non-JSON body for model {self._model!r}: {resp.text[:500]}"
            ) from exc

        content = payload.get("content")
        if not isinstance(content, list):
            raise GenerationError(
                f"Anthropic returned a malformed payload for model {self._model!r}: {payload!r}"
            )
        parts = [
            item.get("text")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        completion = "".join(part for part in parts if isinstance(part, str)).strip()
        if not completion:
            raise GenerationError(f"Anthropic returned an empty completion for {self._model!r}.")
        return completion


class GeminiLLMClient(_PooledHTTPClientMixin):
    """LLM client for Google Gemini ``generateContent``."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 120.0,
        policy: NetworkPolicyGate | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._policy = policy

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        path = f"/v1beta/models/{self._model}:generateContent"
        url = f"{self._base_url}{path}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        headers = {"x-goog-api-key": self._api_key}

        _check_policy_or_raise(
            policy=self._policy,
            base_url=self._base_url,
            path=path,
            source="gemini_llm.generate",
            provider="Gemini",
            default_scheme="https",
        )

        try:
            resp = self._http_client().post(url, json=body, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            raise LLMUnavailableError(
                f"Cannot reach Gemini LLM endpoint at {self._base_url}: {exc}."
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationError(f"Gemini request to {self._base_url} failed: {exc}") from exc

        if resp.status_code >= 400:
            raise GenerationError(
                f"Gemini returned HTTP {resp.status_code} for "
                f"model {self._model!r}: {resp.text[:500]}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise GenerationError(
                f"Gemini returned non-JSON body for model {self._model!r}: {resp.text[:500]}"
            ) from exc

        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise GenerationError(
                f"Gemini returned a malformed payload for model {self._model!r}: {payload!r}"
            )
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise GenerationError(
                f"Gemini returned a malformed candidate for model {self._model!r}."
            )
        completion = "".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        ).strip()
        if not completion:
            raise GenerationError(f"Gemini returned an empty completion for {self._model!r}.")
        return completion


# ---------------------------------------------------------------------------
# Deterministic stub (offline-safe, deterministic, test-friendly)
# ---------------------------------------------------------------------------


#: Regex matching the citation lines the prompt builder emits. Pinned here
#: AND in the prompt builder so a change in either side fails the
#: deterministic-stub tests immediately. Format::
#:
#:     [1] **FUNDED** — Horizon Europe call HORIZON-..., p. 12, §Methodology
#:
#: The leading bracket+digits is the only part the stub actually parses.
_PROMPT_CITATION_LINE = re.compile(r"^\[(\d{1,2})\]\s+\*\*", re.MULTILINE)


class DeterministicLLMClient:
    """Test-only LLM stub that produces deterministic, citation-aware output.

    Strategy: scan the prompt for ``[N]`` citation lines emitted by the
    prompt builder, then synthesize a paragraph that references each
    ``[N]`` with a section-type-aware opening sentence. Same prompt
    twice → identical output, so this client is the natural choice for
    tests that need to assert on exact draft content.

    Used as:

    * The default LLM in every generation-layer unit test (no Ollama
      required in CI).
    * The offline fallback inside :func:`make_llm_client` when Ollama
      is unreachable AND ``offline_mode=True``. This is what keeps the
      AC3 invariant ("Generation can run using a local model runtime
      without cloud API calls") satisfied even on a fresh machine
      without Ollama installed.

    The output quality is intentionally degraded — the stub does not
    interpret the prompt's intent, it just reflects citation markers —
    so a developer who got the stub by accident will notice immediately.
    """

    MODEL_NAME = "deterministic-stub-v1"

    def __init__(self, *, model: str = MODEL_NAME) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,  # noqa: ARG002 — kept to satisfy the protocol
        temperature: float = 0.2,  # noqa: ARG002 — same; deterministic = no temp
    ) -> str:
        """Reflect the prompt's ``[N]`` markers back as a deterministic paragraph.

        ``max_tokens`` and ``temperature`` are accepted for protocol
        compatibility but ignored — the stub is deliberately
        non-stochastic.
        """

        markers = sorted({int(m.group(1)) for m in _PROMPT_CITATION_LINE.finditer(prompt)})

        # Section-type heuristic: the prompt builder writes
        # ``**Section type:** <Title>`` near the top. Pulling the
        # title out lets the stub produce a sentence that names the
        # section, which makes test failures friendlier without
        # adding non-determinism.
        section_match = re.search(r"\*\*Section type:\*\*\s+(.+)", prompt)
        section_title = section_match.group(1).strip() if section_match else "Section"

        if not markers:
            # No citations were retrieved (empty corpus) — still emit a
            # valid, non-empty draft so the workflow can return a
            # well-formed GenerationDraft. The text is intentionally
            # explicit so a downstream reviewer can tell what happened.
            return (
                f"Draft for the {section_title} section. "
                f"No retrieved evidence was available; expand the index "
                f"with relevant past proposals before relying on this draft."
            )

        # One sentence per citation marker keeps the output short and
        # makes the citation-validation logic in the workflow easy to
        # exercise. The phrasing is deliberately neutral; this is a
        # stub, not a real drafter.
        sentences = [f"Draft for the {section_title} section, derived from retrieved evidence."]
        for n in markers:
            sentences.append(
                f"This sentence references retrieved example [{n}] as supporting evidence."
            )
        return " ".join(sentences)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _ollama_llm_reachable(
    base_url: str,
    timeout: float = 2.0,
    *,
    policy: NetworkPolicyGate | None = None,
) -> bool:
    """Best-effort TCP probe of the Ollama daemon's host:port.

    Same shape as :func:`eurpe.retrieval.embeddings._ollama_reachable`
    — a separate function lives here (rather than re-using the
    retrieval helper) so tests can monkeypatch one or the other
    independently. This matters under the ``no_network`` fixture: the
    fixture raises on every ``socket.connect``, and a test that wants
    to exercise the LLM-only fallback path needs to patch *this*
    function specifically.

    When ``policy`` is supplied, the gate is consulted FIRST. A deny
    is treated as "not reachable" (caller falls back to the
    deterministic LLM client) rather than re-raising — the factory's
    contract is to degrade gracefully, not to crash the application.
    The denial is still recorded in the audit log by the gate itself.

    Catches :class:`OSError` (the standard "connection refused / timed
    out / unreachable" family) and returns ``False``. Other exception
    types — most importantly :class:`pytest.fail.Exception` from the
    ``no_network`` fixture — are allowed to propagate. Tests that
    need the factory's fallback path under ``no_network`` should
    monkeypatch this function directly.
    """

    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    if policy is not None:
        try:
            policy.check(
                host=host,
                port=port,
                scheme=parsed.scheme or "tcp",
                path="/",
                source="ollama_llm.reachable_probe",
            )
        except EgressDeniedError:
            return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def make_llm_client(config: object) -> LLMClient:
    """Pick an LLM client for ``config``.

    Strategy:

    * If runtime is ``ollama`` and offline mode is on AND Ollama is unreachable → return
      :class:`DeterministicLLMClient` and log a warning. This is what
      keeps the offline contract intact: tests run without a daemon,
      and developers without Ollama still get a working pipeline
      (with degraded quality).
    * If runtime is ``ollama`` and offline mode is on AND Ollama IS reachable → return
      :class:`OllamaLLMClient`. The user has Ollama; use it.
    * If runtime is one of the OpenAI-compatible engines/providers,
      return the matching HTTP client. Non-loopback requests still go
      through :class:`NetworkPolicyGate`; selecting a cloud provider is
      not enough to bypass the allowlist.
    * If runtime is ``anthropic`` or ``gemini``, return the dedicated
      client and require the provider API key from the configured
      environment variable.

    The ``config`` parameter is typed ``object`` rather than
    ``EurpeConfig`` to avoid a top-level import cycle (the same trick
    :func:`make_embedder` uses). Duck-typed access to
    ``offline_mode``, ``models.llm_model``, and
    ``models.ollama_base_url`` is enough.
    """

    offline_mode = bool(getattr(config, "offline_mode", True))
    models = getattr(config, "models", None)
    runtime = str(getattr(models, "runtime", "ollama")).lower()
    llm_model = getattr(models, "llm_model", "llama3.1:8b")
    ollama_base_url = getattr(models, "ollama_base_url", "http://localhost:11434")
    llm_base_url = getattr(models, "llm_base_url", None)
    api_key_env = getattr(models, "llm_api_key_env", None)

    # Build the gate once and pass it to both the probe and the
    # client. ``make_network_policy`` is duck-typed; a partial mock
    # without ``network_audit_log_path`` would raise — wrap narrowly so
    # only the documented "partial mock" failure modes (ValueError when
    # runtime_dir/audit_log_path are both missing, TypeError on a
    # non-path return, AttributeError on a heavily stubbed mock) fall
    # back to the legacy no-gate path. A real construction bug must
    # surface, not fail-open silently.
    policy: NetworkPolicyGate | None
    try:
        policy = make_network_policy(config)
    except (ValueError, TypeError, AttributeError) as exc:  # pragma: no cover - defensive mock
        if runtime != "ollama":
            if runtime in _OPENAI_COMPATIBLE_DEFAULTS:
                default_base_url, _default_env = _OPENAI_COMPATIBLE_DEFAULTS[runtime]
                candidate_base_url = llm_base_url or default_base_url
            elif runtime == "anthropic":
                candidate_base_url = llm_base_url or _ANTHROPIC_DEFAULT_BASE_URL
            elif runtime == "gemini":
                candidate_base_url = llm_base_url or _GEMINI_DEFAULT_BASE_URL
            else:
                candidate_base_url = ""
            if candidate_base_url and _requires_policy(candidate_base_url):
                raise LLMUnavailableError(
                    f"Cannot construct {runtime} LLM client without a network policy. "
                    "Set config.runtime_dir or config.network_audit_log_path() so "
                    "non-loopback generation can be checked before HTTP requests."
                ) from exc
        policy = None

    if (
        runtime == "ollama"
        and offline_mode
        and not _ollama_llm_reachable(
            ollama_base_url,
            policy=policy,
        )
    ):
        logger.warning(
            "Ollama not reachable at %s and offline_mode is on; "
            "falling back to DeterministicLLMClient. Drafting quality "
            "will be poor — start `ollama serve` and `ollama pull %s` "
            "for real generations.",
            ollama_base_url,
            llm_model,
        )
        return DeterministicLLMClient()

    if runtime == "ollama":
        return OllamaLLMClient(
            base_url=ollama_base_url,
            model=llm_model,
            policy=policy,
        )

    if runtime in _OPENAI_COMPATIBLE_DEFAULTS:
        default_base_url, default_env = _OPENAI_COMPATIBLE_DEFAULTS[runtime]
        env_var = api_key_env or default_env
        api_key = _read_api_key(
            provider=runtime,
            env_var=env_var,
            required=default_env is not None and runtime != "vllm",
        )
        return OpenAICompatibleLLMClient(
            provider=runtime,
            base_url=llm_base_url or default_base_url,
            model=llm_model,
            api_key=api_key,
            policy=policy,
        )

    if runtime == "anthropic":
        env_var = api_key_env or "ANTHROPIC_API_KEY"
        return AnthropicLLMClient(
            base_url=llm_base_url or _ANTHROPIC_DEFAULT_BASE_URL,
            model=llm_model,
            api_key=_read_api_key(provider="anthropic", env_var=env_var, required=True) or "",
            policy=policy,
        )

    if runtime == "gemini":
        env_var = api_key_env or "GEMINI_API_KEY"
        return GeminiLLMClient(
            base_url=llm_base_url or _GEMINI_DEFAULT_BASE_URL,
            model=llm_model,
            api_key=_read_api_key(provider="gemini", env_var=env_var, required=True) or "",
            policy=policy,
        )

    raise GenerationError(f"Unsupported LLM runtime {runtime!r}")
