"""Exception hierarchy for the retrieval package.

A single base (:class:`IndexingError`) so callers — most importantly
:mod:`eurpe.retrieval.cli` — can ``except IndexingError`` once and surface
a clean ``error: ...`` line on every retrieval-layer failure. Mirrors the
``IngestionError`` pattern in :mod:`eurpe.ingestion.errors` so the two
packages feel idiomatic side-by-side.

Subclasses identify *category* rather than *cause*:

* :class:`EmbeddingError` — anything that goes wrong during vector
  production (Ollama unreachable, unknown model name, malformed
  response).
* :class:`OfflineEmbeddingError` — a strict subclass for the case where
  the caller asked for a real embedder under offline mode and no
  fallback is available. Distinct so a future "strict offline" mode can
  branch on it without parsing message strings.

Keeping ``IndexingError`` as the public root mirrors the naming of the
acceptance criterion ("local embedding/index creation works without
outbound network access") — every recoverable failure of that operation
is an indexing-layer failure from the caller's point of view.
"""

from __future__ import annotations


class IndexingError(Exception):
    """Base class for any failure inside :mod:`eurpe.retrieval`.

    Catch this in callers (CLI, API) when you want one branch for "the
    indexing/retrieval layer broke". Subclasses carry more specific
    intent for handlers that care.
    """


class EmbeddingError(IndexingError):
    """Raised when producing an embedding fails.

    Covers connection problems with Ollama, unknown model names, and
    malformed embedding responses. The cause (original exception, if
    any) is attached via ``raise … from …`` so handlers can walk
    ``__cause__`` for the underlying error type.
    """


class OfflineEmbeddingError(EmbeddingError):
    """Offline mode requested but no offline embedder is available.

    This is a fail-fast signal that the configuration asked for a real
    embedder (e.g., ``OllamaEmbedder``) and the offline-fallback path
    was disabled or did not produce a usable instance. ``make_embedder``
    today never raises this directly — it always falls back to
    :class:`~eurpe.retrieval.embeddings.DeterministicHashEmbedder` — but
    a future "strict" mode would throw it instead of falling back.
    """
