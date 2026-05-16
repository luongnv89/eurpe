"""Embedding backends for the retrieval layer.

Two concrete implementations sit behind the :class:`Embedder` Protocol:

* :class:`DeterministicHashEmbedder` — pure-Python, zero-network, zero
  model files. Hashes whitespace-split tokens into a fixed-dimension
  vector and L2-normalises the result. Same input always produces the
  same output, so it is the natural choice for tests and as the
  offline-fallback when no real embedder is reachable.
* :class:`OllamaEmbedder` — calls a local Ollama daemon at
  ``POST /api/embeddings``. The daemon listens on ``localhost`` so
  this is "local" traffic; ``offline_mode`` does NOT block it. The
  embedder still gates itself on a known-model check so a typo in the
  model name fails at construction rather than at the first request.

The :func:`make_embedder` factory picks one based on ``EurpeConfig``:
in offline mode it probes the Ollama TCP port and falls back to the
deterministic hash embedder if the daemon is not running. The
fallback is loud (a warning is logged) so a developer who expected
real semantic retrieval gets a chance to notice.

Why a deterministic hash embedder is appropriate for tests
----------------------------------------------------------
Real embeddings are slow, depend on a model download, and are not bit-
for-bit reproducible across hardware. The acceptance criterion "Index
can be rebuilt from fixtures and queried in a deterministic test"
demands reproducibility, which rules out a real model in CI. The hash
embedder gives us a stable, side-effect-free vector we can compare
across runs. Its retrieval *quality* is intentionally poor — it can
only match on token overlap — but that is enough to verify the
plumbing.
"""

from __future__ import annotations

import hashlib
import logging
import math
import socket
from typing import Protocol
from urllib.parse import urlparse

import httpx

from eurpe.retrieval.errors import EmbeddingError
from eurpe.security import (
    EgressDeniedError,
    NetworkPolicyGate,
    make_network_policy,
)

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Synchronous, batch-friendly embedding interface.

    Concrete implementations MUST be safe to call from a single thread
    in a tight loop; the index batches small lists of texts (typically
    1–32 chunks at a time).
    """

    @property
    def dimension(self) -> int:
        """Width of the produced vectors. Constant for the lifetime of the embedder."""

    @property
    def model_name(self) -> str:
        """Human-readable identifier of the underlying model.

        Recorded in index metadata so a future migration script can
        detect when the embedder behind an existing index changed.
        """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""


# ---------------------------------------------------------------------------
# Deterministic hash embedder (offline-safe, deterministic, test-friendly)
# ---------------------------------------------------------------------------


class DeterministicHashEmbedder:
    """Pure-Python embedder that hashes tokens into a fixed-dim vector.

    Algorithm:

    1. Lowercase the text and split on whitespace.
    2. For each token, take ``sha256(token)``, read the first 8 bytes
       as a big-endian unsigned int, and modulo into the dimension to
       pick an index.
    3. Add ``1 / sqrt(token_count)`` at that index (so longer texts do
       not dominate purely on length).
    4. L2-normalise the resulting vector. Empty texts produce a zero
       vector — they cannot be normalised, so they are returned as-is
       and the index treats them as "no information".

    The result is bit-for-bit reproducible across runs and machines (it
    only uses the standard library), which is what makes it suitable
    for the deterministic-query acceptance criterion. Retrieval
    *quality* is poor — there is no semantic structure — so this MUST
    NOT be used in production unless a real embedder is unreachable
    and a degraded fallback is preferable to a hard error.
    """

    MODEL_NAME = "deterministic-hash-v1"

    def __init__(self, *, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text. Order-preserving."""

        return [self._embed_one(t) for t in texts]

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _embed_one(self, text: str) -> list[float]:
        # ``str.split()`` with no argument collapses any run of
        # whitespace into a single delimiter and drops empties — which
        # is exactly what we want for plain prose. Lowercasing keeps
        # 'Excellence' and 'excellence' in the same bucket so a query
        # is not penalised for matching only on case.
        tokens = text.lower().split()
        vector = [0.0] * self._dimension
        if not tokens:
            return vector
        weight = 1.0 / math.sqrt(len(tokens))
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:8], "big") % self._dimension
            vector[idx] += weight
        # L2-normalise so cosine similarity is well-defined and bounded.
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:  # pragma: no cover - all-zero vector path
            return vector
        return [v / norm for v in vector]


# ---------------------------------------------------------------------------
# Ollama embedder (real model, talks to localhost)
# ---------------------------------------------------------------------------


class OllamaEmbedder:
    """Embedder backed by a local Ollama daemon (``POST /api/embeddings``).

    Talks to ``localhost:11434`` by default, which is what a developer
    machine running ``ollama serve`` exposes. ``offline_mode`` does
    NOT block this — Ollama is a local process; ``offline_mode``'s
    invariant is "no outbound *Internet* traffic" — but the
    :func:`make_embedder` factory will fall back to
    :class:`DeterministicHashEmbedder` if Ollama is not reachable, so
    callers in stricter environments still get a working embedder.

    Dimension is hard-coded per-model in :attr:`KNOWN_DIMS` rather than
    discovered at runtime. Discovering would require either an extra
    request or storing a dimension-per-model file in the index; the
    Ollama embedding endpoint does not return the dimension in its
    response. Hard-coding catches typos at construction time (raising
    :class:`EmbeddingError`) which is friendlier than failing later
    inside Chroma when an embedding of the wrong width is upserted.
    """

    #: Public, well-known embedding model dimensions. Update when adding a new model.
    KNOWN_DIMS: dict[str, int] = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        policy: NetworkPolicyGate | None = None,
    ) -> None:
        if model not in self.KNOWN_DIMS:
            # Surface this as :class:`EmbeddingError` not ``ValueError``
            # so callers that already wrap ``IndexingError`` catch it
            # in the same branch as runtime embedding failures.
            raise EmbeddingError(
                f"Unknown embedding model {model!r}; "
                f"add its dimension to OllamaEmbedder.KNOWN_DIMS first. "
                f"Known: {sorted(self.KNOWN_DIMS)}"
            )
        self._model = model
        self._dimension = self.KNOWN_DIMS[model]
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # ``policy`` is optional and defaults to ``None`` so existing
        # tests that build an ``OllamaEmbedder`` directly (without
        # going through ``make_embedder``) keep working. The factory
        # always wires the gate; only production paths consult it.
        self._policy = policy

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text via one ``/api/embeddings`` POST per item.

        Ollama's embeddings endpoint accepts a single ``prompt`` per
        call, so we loop. The cost is dominated by the model on the
        Ollama side, not by HTTP overhead, so micro-batching would not
        help meaningfully. Using one ``httpx.Client`` for the whole
        batch keeps the connection alive across calls.
        """

        out: list[list[float]] = []
        url = f"{self._base_url}/api/embeddings"
        # Gate the first thing in the request lifecycle so a deny
        # raises BEFORE the httpx Client is even constructed. The
        # check is per-batch (not per-text) because the host:port and
        # scheme are constant for the lifetime of the embedder.
        if self._policy is not None:
            parsed = urlparse(self._base_url)
            self._policy.check(
                host=parsed.hostname or "localhost",
                port=parsed.port or 11434,
                scheme=parsed.scheme or "http",
                path="/api/embeddings",
                source="ollama_embedder.embed",
            )
        try:
            with httpx.Client(timeout=self._timeout) as client:
                for text in texts:
                    resp = client.post(url, json={"model": self._model, "prompt": text})
                    resp.raise_for_status()
                    payload = resp.json()
                    embedding = payload.get("embedding")
                    if not isinstance(embedding, list) or not embedding:
                        raise EmbeddingError(
                            f"Ollama returned a malformed embedding payload: {payload!r}"
                        )
                    if len(embedding) != self._dimension:
                        raise EmbeddingError(
                            f"Embedding dim mismatch for model {self._model!r}: "
                            f"got {len(embedding)}, expected {self._dimension}. "
                            "Did the model identifier change on the Ollama side?"
                        )
                    out.append([float(v) for v in embedding])
        except httpx.HTTPError as exc:
            # ``httpx.HTTPError`` is the umbrella covering connect /
            # timeout / status errors. Wrap in :class:`EmbeddingError`
            # so the CLI / API can present a uniform message and so
            # ``except IndexingError`` works.
            raise EmbeddingError(f"Ollama request failed: {exc}") from exc
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _ollama_reachable(
    base_url: str,
    timeout: float = 2.0,
    *,
    policy: NetworkPolicyGate | None = None,
) -> bool:
    """Best-effort TCP probe of the Ollama daemon's host:port.

    Cheap and fast: we open a TCP connection and immediately close it.
    Failing fast (TCP-level) avoids a multi-second HTTP timeout on
    machines where Ollama isn't running.

    When ``policy`` is supplied, the gate is consulted FIRST. A deny
    is treated as "not reachable" (caller falls back to the
    deterministic embedder) rather than re-raising — the factory's
    contract is to degrade gracefully, not to crash the application.
    The denial is still recorded in the audit log by the gate itself.

    Catches :class:`OSError` (the standard "connection refused / timed
    out / unreachable" family) and returns ``False``. Any other
    exception type is allowed to propagate — in particular,
    :class:`pytest.fail.Exception` (a ``BaseException``) raised by the
    project's ``no_network`` fixture is not caught here. Tests that
    need the factory's fallback path under ``no_network`` should
    monkeypatch this function directly rather than relying on the
    probe to suppress the fixture's signal.
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
                source="ollama_embedder.reachable_probe",
            )
        except EgressDeniedError:
            # Treat a denied probe the same way as a refused TCP
            # connection: the factory will fall back to the
            # deterministic embedder. The denial is already in the
            # audit log via gate.check().
            return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def make_embedder(config: object) -> Embedder:
    """Pick an embedder for ``config``.

    Strategy:

    * If offline mode is on AND Ollama is unreachable → return
      :class:`DeterministicHashEmbedder` and log a warning. This is
      what keeps the offline contract intact: tests run without a
      daemon, and developers without Ollama still get a working
      pipeline (with degraded retrieval quality).
    * Otherwise build :class:`OllamaEmbedder` with the configured
      model and base URL. We do NOT probe Ollama in non-offline mode
      because the user explicitly asked for a real embedder; surfacing
      a connection error at the first ``embed()`` call gives them a
      clear message about which call site failed.

    The ``config`` parameter is typed ``object`` rather than
    ``EurpeConfig`` to avoid a top-level import cycle
    (``eurpe.config`` does not depend on ``eurpe.retrieval`` and
    importing it here would invert that). Duck-typed access to the
    fields we need below is enough.
    """

    # Duck-type: fields used are ``offline_mode`` and ``models.{embedding_model,
    # ollama_base_url}``. Using ``getattr`` keeps this importable even in
    # hypothetical contexts where the caller passes a partial mock.
    offline_mode = bool(getattr(config, "offline_mode", True))
    models = getattr(config, "models", None)
    embedding_model = getattr(models, "embedding_model", "nomic-embed-text")
    ollama_base_url = getattr(models, "ollama_base_url", "http://localhost:11434")

    # Build the network policy once and pass it to both the probe and
    # the embedder. ``make_network_policy`` is duck-typed on the same
    # config so a partial mock still works. Narrow except: only the
    # documented "partial mock" failure modes (missing runtime_dir →
    # ValueError, non-path audit_log_path → TypeError, missing attribute
    # on a heavily stubbed mock → AttributeError) fall through to the
    # legacy no-gate path. A real bug in NetworkPolicyGate construction
    # (anything else) MUST surface loudly rather than silently fail-open.
    policy: NetworkPolicyGate | None
    try:
        policy = make_network_policy(config)
    except (ValueError, TypeError, AttributeError):  # pragma: no cover - defensive: degraded mock
        policy = None

    if offline_mode and not _ollama_reachable(ollama_base_url, policy=policy):
        logger.warning(
            "Ollama not reachable at %s and offline_mode is on; "
            "falling back to DeterministicHashEmbedder. Retrieval quality "
            "will be poor — start `ollama serve` for real embeddings.",
            ollama_base_url,
        )
        return DeterministicHashEmbedder(dimension=384)

    return OllamaEmbedder(
        model=embedding_model,
        base_url=ollama_base_url,
        policy=policy,
    )
