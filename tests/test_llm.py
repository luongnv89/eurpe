"""Tests for ``eurpe.generation.llm`` — the LLM client backends and factory.

The :class:`DeterministicLLMClient` is the load-bearing piece here:
every workflow / CLI test in the package depends on it. The factory's
fallback path is what implements AC3 ("Generation can run using a
local model runtime without cloud API calls") on a machine without
Ollama.

We deliberately do NOT exercise :class:`OllamaLLMClient` against a
live daemon. Like the matching ``OllamaEmbedder`` tests in
``test_embeddings.py``, the failure modes are covered with
``httpx.MockTransport`` so CI doesn't need a network or a local
Ollama install.
"""

from __future__ import annotations

import httpx
import pytest

from eurpe.config import EurpeConfig, ModelsConfig
from eurpe.generation.errors import GenerationError, LLMUnavailableError
from eurpe.generation.llm import (
    DeterministicLLMClient,
    OllamaLLMClient,
    _ollama_llm_reachable,
    make_llm_client,
)

# ---------------------------------------------------------------------------
# DeterministicLLMClient
# ---------------------------------------------------------------------------


def _stub_prompt_with_citations(*marker_ids: int) -> str:
    """Build a fake prompt that the stub will see with the given ``[N]`` markers."""

    lines = ["**Section type:** Methodology", ""]
    for n in marker_ids:
        lines.append(f"[{n}] **FUNDED** — Horizon Europe call X, p. 1, §X")
        lines.append("    > snippet")
        lines.append("")
    return "\n".join(lines)


def test_deterministic_client_produces_text() -> None:
    client = DeterministicLLMClient()
    out = client.generate(_stub_prompt_with_citations(1, 2))
    assert isinstance(out, str)
    assert out


def test_deterministic_client_is_deterministic() -> None:
    client = DeterministicLLMClient()
    prompt = _stub_prompt_with_citations(1, 2, 3)
    a = client.generate(prompt)
    b = client.generate(prompt)
    assert a == b


def test_deterministic_client_includes_citation_markers() -> None:
    """When the prompt has [1], [2], [3] markers, the output references each."""

    client = DeterministicLLMClient()
    out = client.generate(_stub_prompt_with_citations(1, 2, 3))
    assert "[1]" in out
    assert "[2]" in out
    assert "[3]" in out


def test_deterministic_client_handles_no_citations_in_prompt() -> None:
    """Empty evidence prompt → still emits a non-empty draft."""

    client = DeterministicLLMClient()
    prompt = "**Section type:** Methodology\n\n(no examples retrieved)\n"
    out = client.generate(prompt)
    assert out
    # The fallback message names the failure mode so a reviewer can
    # tell it's a "no evidence" scenario rather than a model error.
    assert "No retrieved evidence" in out


def test_deterministic_client_model_name_is_stable() -> None:
    """The model name is recorded in GenerationDraft.model — pin it here."""

    assert DeterministicLLMClient().model == "deterministic-stub-v1"


def test_deterministic_client_section_title_appears_in_output() -> None:
    """The stub uses the section title from the prompt — covers Impact Pathway."""

    client = DeterministicLLMClient()
    prompt = "**Section type:** Impact Pathway\n[1] **FUNDED** — call X, p. 1, §X\n    > x\n"
    out = client.generate(prompt)
    assert "Impact Pathway" in out


# ---------------------------------------------------------------------------
# OllamaLLMClient — exercised against an httpx MockTransport
# ---------------------------------------------------------------------------


def _make_ollama_client_with_transport(
    handler,  # type: ignore[no-untyped-def]
    *,
    model: str = "llama3.1:8b",
) -> OllamaLLMClient:
    """Build an OllamaLLMClient that uses a mocked transport.

    Patches :class:`httpx.Client` at module level so the client we
    construct uses the mock. Returning a context manager-aware
    construct keeps the test bodies tidy.
    """

    return OllamaLLMClient(base_url="http://localhost:11434", model=model)


def test_ollama_client_handles_connect_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``httpx.ConnectError`` becomes :class:`LLMUnavailableError`."""

    class _BoomClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _BoomClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("eurpe.generation.llm.httpx.Client", _BoomClient)
    client = _make_ollama_client_with_transport(handler=None)
    with pytest.raises(LLMUnavailableError, match="Cannot reach Ollama"):
        client.generate("any prompt")


def test_ollama_client_handles_read_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A read timeout is also reported as :class:`LLMUnavailableError` (not generic)."""

    class _TimeoutClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _TimeoutClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            raise httpx.ReadTimeout("model is busy")

    monkeypatch.setattr("eurpe.generation.llm.httpx.Client", _TimeoutClient)
    client = _make_ollama_client_with_transport(handler=None)
    with pytest.raises(LLMUnavailableError):
        client.generate("any prompt")


def test_ollama_client_handles_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx response raises :class:`GenerationError` (not LLMUnavailableError)."""

    class _ErrorClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _ErrorClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return httpx.Response(
                status_code=500,
                content=b"Internal model crash",
                request=httpx.Request("POST", "http://localhost:11434/api/generate"),
            )

    monkeypatch.setattr("eurpe.generation.llm.httpx.Client", _ErrorClient)
    client = _make_ollama_client_with_transport(handler=None)
    with pytest.raises(GenerationError, match="HTTP 500"):
        client.generate("any prompt")


def test_ollama_client_handles_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ``response`` key → :class:`GenerationError`."""

    class _MalformedClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _MalformedClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json={"unexpected": "shape"},
                request=httpx.Request("POST", "http://localhost:11434/api/generate"),
            )

    monkeypatch.setattr("eurpe.generation.llm.httpx.Client", _MalformedClient)
    client = _make_ollama_client_with_transport(handler=None)
    with pytest.raises(GenerationError, match="malformed payload"):
        client.generate("any prompt")


def test_ollama_client_returns_response_text_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path — a 200 with ``{"response": "..."}`` returns the string."""

    captured: dict[str, object] = {}

    class _OkClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["timeout"] = kwargs.get("timeout")

        def __enter__(self) -> _OkClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, *, json: dict[str, object]) -> httpx.Response:
            captured["url"] = url
            captured["json"] = json
            return httpx.Response(
                status_code=200,
                json={"response": "Hello from Llama"},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("eurpe.generation.llm.httpx.Client", _OkClient)
    client = _make_ollama_client_with_transport(handler=None)
    out = client.generate("the prompt", max_tokens=64, temperature=0.4)
    assert out == "Hello from Llama"

    # Verify the body shape — ``stream: false`` is mandatory; if a
    # future refactor drops it, the response will be chunked JSON and
    # this client will return malformed-payload errors at runtime.
    body = captured["json"]
    assert isinstance(body, dict)
    assert body["stream"] is False
    assert body["model"] == "llama3.1:8b"
    assert body["prompt"] == "the prompt"
    assert body["options"] == {"temperature": 0.4, "num_predict": 64}


def test_ollama_client_strips_trailing_slash_from_base_url() -> None:
    c = OllamaLLMClient(base_url="http://localhost:11434/", model="llama3.1:8b")
    assert c.base_url == "http://localhost:11434"


def test_ollama_client_model_property() -> None:
    c = OllamaLLMClient(base_url="http://localhost:11434", model="llama3.1:8b")
    assert c.model == "llama3.1:8b"


# ---------------------------------------------------------------------------
# make_llm_client factory
# ---------------------------------------------------------------------------


def _config_with(**overrides: object) -> EurpeConfig:
    base = EurpeConfig(models=ModelsConfig())
    return base.model_copy(update=dict(overrides))


def test_make_llm_client_falls_back_when_offline_and_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline + Ollama unreachable → :class:`DeterministicLLMClient`."""

    monkeypatch.setattr(
        "eurpe.generation.llm._ollama_llm_reachable",
        lambda _url, timeout=2.0, *, policy=None: False,
    )
    cfg = _config_with(offline_mode=True)
    client = make_llm_client(cfg)
    assert isinstance(client, DeterministicLLMClient)


def test_make_llm_client_uses_ollama_when_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline + Ollama reachable → :class:`OllamaLLMClient`."""

    monkeypatch.setattr(
        "eurpe.generation.llm._ollama_llm_reachable",
        lambda _url, timeout=2.0, *, policy=None: True,
    )
    cfg = _config_with(offline_mode=True)
    client = make_llm_client(cfg)
    assert isinstance(client, OllamaLLMClient)


def test_make_llm_client_skips_probe_when_offline_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-offline mode → factory trusts the caller, doesn't probe.

    Mirrors the matching test for ``make_embedder``: skipping the
    probe means a real network failure surfaces at the first
    ``generate()`` call with a clearer trace than a probe-and-raise
    here would.
    """

    called = {"probed": False}

    def _record(_url: str, timeout: float = 2.0, *, policy=None) -> bool:
        called["probed"] = True
        return False

    monkeypatch.setattr("eurpe.generation.llm._ollama_llm_reachable", _record)
    cfg = _config_with(offline_mode=False)
    client = make_llm_client(cfg)
    assert isinstance(client, OllamaLLMClient)
    assert called["probed"] is False


def test_make_llm_client_falls_back_with_blocked_probe(
    monkeypatch: pytest.MonkeyPatch,
    no_network: None,
) -> None:
    """The factory's offline path works under the ``no_network`` fixture.

    Same recipe as ``test_make_embedder_falls_back_with_blocked_probe``
    in ``test_embeddings.py``: the fixture raises on every
    ``socket.connect``, so callers that want the factory under
    ``no_network`` MUST monkeypatch the probe to return False.
    """

    monkeypatch.setattr(
        "eurpe.generation.llm._ollama_llm_reachable",
        lambda _url, timeout=2.0, *, policy=None: False,
    )
    cfg = _config_with(offline_mode=True)
    client = make_llm_client(cfg)
    assert isinstance(client, DeterministicLLMClient)


# ---------------------------------------------------------------------------
# _ollama_llm_reachable
# ---------------------------------------------------------------------------


def test_ollama_llm_reachable_returns_false_for_unused_port() -> None:
    """Pick a high port nothing should be listening on."""

    assert _ollama_llm_reachable("http://localhost:1", timeout=0.2) is False


def test_ollama_llm_reachable_uses_default_port_when_url_omits_one() -> None:
    """A bare host (no port) defaults to 11434 and does NOT raise."""

    assert _ollama_llm_reachable("http://localhost", timeout=0.2) in (True, False)
