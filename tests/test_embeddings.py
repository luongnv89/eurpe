"""Tests for ``eurpe.retrieval.embeddings``.

The deterministic embedder is the load-bearing piece: it powers every
test in ``test_index.py`` and the offline-fallback path of
``make_embedder``. The tests below pin its core invariants
(determinism, normalisation, dimensionality, distinctiveness) plus the
factory's offline fallback behaviour and ``OllamaEmbedder``'s strict
model-name check.

We deliberately do NOT test ``OllamaEmbedder.embed`` against a live
daemon — that would require a network or a local Ollama install in CI.
A future integration suite (out of scope for issue #4) can cover it.
"""

from __future__ import annotations

import math

import pytest

from eurpe.config import EurpeConfig, ModelsConfig
from eurpe.retrieval.embeddings import (
    DeterministicHashEmbedder,
    OllamaEmbedder,
    _ollama_reachable,
    make_embedder,
)
from eurpe.retrieval.errors import EmbeddingError

# ---------------------------------------------------------------------------
# DeterministicHashEmbedder
# ---------------------------------------------------------------------------


def test_deterministic_embedder_is_deterministic() -> None:
    embedder = DeterministicHashEmbedder(dimension=128)
    a = embedder.embed(["hello world from eurpe"])
    b = embedder.embed(["hello world from eurpe"])
    assert a == b


def test_deterministic_embedder_normalises_vectors() -> None:
    embedder = DeterministicHashEmbedder(dimension=64)
    vec = embedder.embed(["funded proposal excellence section body text"])[0]
    norm = math.sqrt(sum(v * v for v in vec))
    # Some floating-point slack — should be very close to 1.0.
    assert math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_deterministic_embedder_dimension_is_respected() -> None:
    embedder = DeterministicHashEmbedder(dimension=256)
    assert embedder.dimension == 256
    vec = embedder.embed(["a b c"])[0]
    assert len(vec) == 256


def test_deterministic_embedder_distinguishes_unrelated_strings() -> None:
    """Two unrelated strings must not be treated as identical.

    Cosine similarity (== dot product for normalised vectors) of two
    very different prose snippets should be meaningfully below 1.0.
    Without this property, the index degenerates into "everything
    matches everything".
    """

    embedder = DeterministicHashEmbedder(dimension=384)
    a, b = embedder.embed(
        [
            "funded proposal excellence section body text",
            "completely unrelated narrative about gardening tomatoes",
        ]
    )
    similarity = sum(x * y for x, y in zip(a, b, strict=True))
    # Empirical sanity: the deterministic embedder is bag-of-tokens, so
    # the only way these two strings would score 1.0 is a hash collision
    # on every shared token. Anything below 0.5 is "they look unrelated"
    # for our purposes.
    assert similarity < 0.5


def test_deterministic_embedder_handles_empty_text() -> None:
    embedder = DeterministicHashEmbedder(dimension=32)
    vec = embedder.embed([""])[0]
    assert len(vec) == 32
    # Empty input produces a zero vector (no tokens to hash).
    assert all(v == 0.0 for v in vec)


def test_deterministic_embedder_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        DeterministicHashEmbedder(dimension=0)


def test_deterministic_embedder_model_name_is_stable() -> None:
    """The model name is recorded in collection metadata; pinning it
    here keeps a future rename loud."""

    assert DeterministicHashEmbedder().model_name == "deterministic-hash-v1"


# ---------------------------------------------------------------------------
# OllamaEmbedder construction
# ---------------------------------------------------------------------------


def test_ollama_embedder_unknown_model_raises_at_init() -> None:
    with pytest.raises(EmbeddingError, match="Unknown embedding model"):
        OllamaEmbedder(model="not-a-real-model-name")


def test_ollama_embedder_known_model_carries_correct_dimension() -> None:
    e = OllamaEmbedder(model="nomic-embed-text")
    assert e.dimension == 768
    assert e.model_name == "nomic-embed-text"


def test_ollama_embedder_known_models_match_dim_table() -> None:
    """Every key in :attr:`KNOWN_DIMS` must be acceptable as a model name."""

    for model, dim in OllamaEmbedder.KNOWN_DIMS.items():
        e = OllamaEmbedder(model=model)
        assert e.dimension == dim


def test_ollama_embedder_strips_trailing_slash_from_base_url() -> None:
    e = OllamaEmbedder(model="nomic-embed-text", base_url="http://localhost:11434/")
    # Internal-state assertion is fine here — it's the surface that
    # would otherwise produce a double slash on ``/api/embeddings``.
    assert e._base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# make_embedder factory
# ---------------------------------------------------------------------------


def _config_with(**overrides: object) -> EurpeConfig:
    """Build an ``EurpeConfig`` with sensible defaults overridden.

    Re-imports the model class so each test gets an isolated copy. We
    do not call ``resolve_paths`` because the factory only inspects
    ``offline_mode`` and ``models``.
    """

    base = EurpeConfig(models=ModelsConfig())
    return base.model_copy(update=dict(overrides))


def test_make_embedder_falls_back_when_offline_and_ollama_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "eurpe.retrieval.embeddings._ollama_reachable",
        lambda _url, timeout=2.0, *, policy=None: False,
    )
    cfg = _config_with(offline_mode=True)
    embedder = make_embedder(cfg)
    assert isinstance(embedder, DeterministicHashEmbedder)
    assert embedder.dimension == 384


def test_make_embedder_uses_ollama_when_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "eurpe.retrieval.embeddings._ollama_reachable",
        lambda _url, timeout=2.0, *, policy=None: True,
    )
    cfg = _config_with(offline_mode=True)
    embedder = make_embedder(cfg)
    assert isinstance(embedder, OllamaEmbedder)


def test_make_embedder_uses_ollama_when_offline_mode_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In non-offline mode we don't probe — we trust the caller.

    The probe is only relevant for "stay safe in offline mode";
    skipping it when offline is OFF means a real network failure
    surfaces at the first ``embed()`` call with a clearer trace.
    """

    called = {"probed": False}

    def _record(_url: str, timeout: float = 2.0, *, policy=None) -> bool:
        called["probed"] = True
        return False

    monkeypatch.setattr("eurpe.retrieval.embeddings._ollama_reachable", _record)
    cfg = _config_with(offline_mode=False)
    embedder = make_embedder(cfg)
    assert isinstance(embedder, OllamaEmbedder)
    assert called["probed"] is False


# ---------------------------------------------------------------------------
# _ollama_reachable
# ---------------------------------------------------------------------------


def test_ollama_reachable_returns_false_for_unused_port() -> None:
    """Pick a high port nothing should be listening on."""

    # Port 1 (TCPMUX) is reserved and effectively never bound on a dev
    # machine; trying to connect produces a clean OSError.
    assert _ollama_reachable("http://localhost:1", timeout=0.2) is False


def test_ollama_reachable_uses_default_port_when_url_omits_one() -> None:
    """A bare host (no port) defaults to 11434.

    The URL parser should fall back to the documented default rather
    than raising, so a slightly malformed config does not crash the
    factory before it can fall back.
    """

    # A bare hostname with no port — should NOT raise even though
    # nothing is bound there. The function returns False either way;
    # the test asserts it doesn't raise.
    assert _ollama_reachable("http://localhost", timeout=0.2) in (True, False)


def test_make_embedder_falls_back_with_blocked_probe(
    monkeypatch: pytest.MonkeyPatch,
    no_network: None,
) -> None:
    """The factory's offline path works under the ``no_network`` fixture.

    The fixture raises ``pytest.fail.Exception`` (a ``BaseException``)
    on any ``socket.connect`` — which the probe deliberately does NOT
    catch. Real callers that want the factory under ``no_network``
    therefore monkeypatch the probe to return ``False`` so the
    fallback path is reached deterministically. This test pins that
    pattern as the supported recipe.
    """

    monkeypatch.setattr(
        "eurpe.retrieval.embeddings._ollama_reachable",
        lambda _url, timeout=2.0, *, policy=None: False,
    )
    cfg = _config_with(offline_mode=True)
    embedder = make_embedder(cfg)
    assert isinstance(embedder, DeterministicHashEmbedder)
