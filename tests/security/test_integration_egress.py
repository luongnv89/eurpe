"""Integration tests: the four egress sites consult the gate.

The four sites:

1. ``eurpe.retrieval.embeddings.OllamaEmbedder.embed`` → POST to
   ``/api/embeddings``.
2. ``eurpe.retrieval.embeddings._ollama_reachable`` → TCP probe used
   by ``make_embedder`` to choose the fallback.
3. ``eurpe.generation.llm.OllamaLLMClient.generate`` → POST to
   ``/api/generate``.
4. ``eurpe.generation.llm._ollama_llm_reachable`` → TCP probe used by
   ``make_llm_client``.

The tests stub the would-be transport calls (httpx.Client and
socket.create_connection) so that if the gate failed to intervene
ahead of them, the stubs would be exercised — assertions on the
stubs detect that regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eurpe.generation.llm import OllamaLLMClient, _ollama_llm_reachable
from eurpe.retrieval.embeddings import OllamaEmbedder, _ollama_reachable
from eurpe.security import AllowlistEntry, EgressDeniedError, NetworkPolicyGate
from eurpe.security.audit import _reset_handlers_for_tests


@pytest.fixture(autouse=True)
def _clean_audit_handlers() -> None:
    _reset_handlers_for_tests()
    yield
    _reset_handlers_for_tests()


@pytest.fixture
def audit_log(tmp_path: Path) -> Path:
    return tmp_path / "audit.log"


@pytest.fixture
def loopback_gate(audit_log: Path) -> NetworkPolicyGate:
    """Gate with no allowlist; only loopback is allowed."""

    return NetworkPolicyGate(allowlist=[], audit_log_path=audit_log)


# ---------------------------------------------------------------------------
# OllamaEmbedder.embed
# ---------------------------------------------------------------------------


def test_ollama_embedder_consults_gate_before_httpx(
    loopback_gate: NetworkPolicyGate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the gate denies, httpx.Client must never be constructed."""

    calls: list[str] = []

    class _ShouldNotBuild:
        def __init__(self, *_, **__) -> None:
            calls.append("Client.__init__")
            raise AssertionError(
                "httpx.Client must not be built when the gate denies"
            )

    import httpx

    monkeypatch.setattr(httpx, "Client", _ShouldNotBuild)

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        base_url="http://example.com:443",  # not loopback, not allowlisted
        policy=loopback_gate,
    )
    with pytest.raises(EgressDeniedError):
        embedder.embed(["some text"])
    assert calls == []


def test_ollama_embedder_allows_loopback_via_gate(
    loopback_gate: NetworkPolicyGate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loopback target should pass through the gate to the transport."""

    transport_built: list[bool] = []

    class _FakeResp:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            # Match the 768-d nomic-embed-text dimension.
            return {"embedding": [0.0] * 768}

    class _FakeClient:
        def __init__(self, *_, **__) -> None:
            transport_built.append(True)

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *args, **kwargs) -> None:
            return None

        def post(self, *_, **__) -> _FakeResp:
            return _FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        policy=loopback_gate,
    )
    out = embedder.embed(["some text"])
    assert transport_built == [True]
    assert len(out) == 1
    assert len(out[0]) == 768


def test_ollama_embedder_allows_allowlisted_host(
    audit_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = NetworkPolicyGate(
        allowlist=[
            AllowlistEntry(host="example.com", port=443, reason="approved mirror")
        ],
        audit_log_path=audit_log,
    )

    transport_built: list[bool] = []

    class _FakeResp:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"embedding": [0.1] * 768}

    class _FakeClient:
        def __init__(self, *_, **__) -> None:
            transport_built.append(True)

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *args, **kwargs) -> None:
            return None

        def post(self, *_, **__) -> _FakeResp:
            return _FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        base_url="https://example.com:443",
        policy=gate,
    )
    embedder.embed(["x"])
    assert transport_built == [True]


# ---------------------------------------------------------------------------
# _ollama_reachable probe
# ---------------------------------------------------------------------------


def test_ollama_reachable_returns_false_on_deny_without_socket(
    loopback_gate: NetworkPolicyGate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe must short-circuit BEFORE socket.create_connection.

    If the gate denies, the function must return False instead of
    propagating EgressDeniedError — that's the contract that lets
    ``make_embedder`` degrade gracefully to the deterministic fallback.
    """

    import socket

    socket_calls: list[bool] = []

    def _should_not_connect(*_, **__) -> None:
        socket_calls.append(True)
        raise AssertionError("socket.create_connection must not be called")

    monkeypatch.setattr(socket, "create_connection", _should_not_connect)

    result = _ollama_reachable("http://example.com:443", policy=loopback_gate)
    assert result is False
    assert socket_calls == []


def test_ollama_reachable_consults_socket_when_allowed(
    loopback_gate: NetworkPolicyGate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the gate allows (loopback), the probe attempts a real connect."""

    import socket

    socket_calls: list[tuple] = []

    class _FakeSocket:
        def __enter__(self) -> _FakeSocket:
            return self

        def __exit__(self, *args, **kwargs) -> None:
            return None

    def _fake_connect(addr, timeout):
        socket_calls.append((addr, timeout))
        return _FakeSocket()

    monkeypatch.setattr(socket, "create_connection", _fake_connect)
    assert _ollama_reachable("http://localhost:11434", policy=loopback_gate) is True
    assert socket_calls == [(("localhost", 11434), 2.0)]


# ---------------------------------------------------------------------------
# OllamaLLMClient.generate
# ---------------------------------------------------------------------------


def test_ollama_llm_consults_gate_before_httpx(
    loopback_gate: NetworkPolicyGate, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class _ShouldNotBuild:
        def __init__(self, *_, **__) -> None:
            calls.append("Client.__init__")
            raise AssertionError(
                "httpx.Client must not be built when the gate denies"
            )

    import httpx

    monkeypatch.setattr(httpx, "Client", _ShouldNotBuild)

    client = OllamaLLMClient(
        base_url="http://example.com:443",
        model="llama3.1:8b",
        policy=loopback_gate,
    )
    with pytest.raises(EgressDeniedError):
        client.generate("prompt-text")
    assert calls == []


def test_ollama_llm_audit_log_excludes_prompt(
    loopback_gate: NetworkPolicyGate, monkeypatch: pytest.MonkeyPatch, audit_log: Path
) -> None:
    """A deny path MUST NOT write the prompt anywhere in the audit log."""

    secret_prompt = "TOP-SECRET-PROPOSAL-CONTENT-SHOULD-NOT-LEAK"

    import httpx

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not build")),
    )

    client = OllamaLLMClient(
        base_url="http://example.com:443",
        model="llama3.1:8b",
        policy=loopback_gate,
    )
    with pytest.raises(EgressDeniedError):
        client.generate(secret_prompt)
    content = audit_log.read_text(encoding="utf-8")
    assert secret_prompt not in content
    assert "TOP-SECRET" not in content


# ---------------------------------------------------------------------------
# _ollama_llm_reachable probe
# ---------------------------------------------------------------------------


def test_ollama_llm_reachable_returns_false_on_deny(
    loopback_gate: NetworkPolicyGate, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("socket must not be touched on deny")
        ),
    )
    assert _ollama_llm_reachable("http://example.com:443", policy=loopback_gate) is False


def test_no_policy_means_backward_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    """OllamaEmbedder built without a policy bypasses the gate entirely.

    This is what every existing test relies on — no implicit network
    call, but no gate either. Behaviour identical to pre-#12 code.
    """

    import httpx

    invoked: list[bool] = []

    class _FakeResp:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"embedding": [0.0] * 768}

    class _FakeClient:
        def __init__(self, *_, **__) -> None:
            invoked.append(True)

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *args, **kwargs) -> None:
            return None

        def post(self, *_, **__) -> _FakeResp:
            return _FakeResp()

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        base_url="http://example.com:443",  # not loopback, but no gate
    )
    embedder.embed(["x"])
    assert invoked == [True]


# ---------------------------------------------------------------------------
# Factory wiring (make_embedder / make_llm_client)
# ---------------------------------------------------------------------------


def test_make_embedder_wires_policy_into_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """make_embedder MUST pass the gate it built into the OllamaEmbedder.

    A regression where the factory builds a gate but forgets to pass
    ``policy=policy`` to ``OllamaEmbedder(...)`` would leave the
    embedder un-gated. This test pins the wiring by configuring a
    non-loopback URL with no allowlist and asserting the embedder's
    first call raises EgressDeniedError.
    """

    from eurpe.config import EurpeConfig, ModelsConfig
    from eurpe.retrieval.embeddings import make_embedder

    # offline_mode=False so the factory skips the probe and returns the
    # real OllamaEmbedder we want to verify is gated.
    cfg = EurpeConfig(
        runtime_dir=tmp_path,
        offline_mode=False,
        models=ModelsConfig(ollama_base_url="http://example.com:443"),
    )

    # Stub httpx.Client so a wiring regression that allowed the call
    # to reach the transport would still not actually hit the network.
    import httpx

    def _should_not_build(*_args, **_kwargs):
        raise AssertionError("httpx.Client must not be built when gate denies")

    monkeypatch.setattr(httpx, "Client", _should_not_build)

    embedder = make_embedder(cfg)
    with pytest.raises(EgressDeniedError):
        embedder.embed(["x"])


def test_make_llm_client_wires_policy_into_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """make_llm_client MUST pass the gate into the OllamaLLMClient.

    Same regression vector as the embedder factory: a refactor that
    forgets to pass ``policy=policy`` would silently un-gate the LLM.
    """

    from eurpe.config import EurpeConfig, ModelsConfig
    from eurpe.generation.llm import make_llm_client

    cfg = EurpeConfig(
        runtime_dir=tmp_path,
        offline_mode=False,
        models=ModelsConfig(
            ollama_base_url="http://example.com:443",
            llm_model="llama3.1:8b",
        ),
    )

    import httpx

    def _should_not_build(*_args, **_kwargs):
        raise AssertionError("httpx.Client must not be built when gate denies")

    monkeypatch.setattr(httpx, "Client", _should_not_build)

    client = make_llm_client(cfg)
    with pytest.raises(EgressDeniedError):
        client.generate("any-prompt")
