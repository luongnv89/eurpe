"""Retrieval package for EURPE.

Hosts the hierarchical chunker, the embedder protocol + concrete
backends, and the local Chroma vector index. Public surface kept
narrow on purpose so the rest of the codebase imports from
``eurpe.retrieval`` rather than internal modules:

* :class:`Chunk` — what the chunker produces and the index consumes.
* :class:`HierarchicalChunker` — splits a :class:`ParsedProposal` plus
  its :class:`ProposalMetadata` into chunks with full provenance.
* :class:`Embedder` (protocol), :class:`DeterministicHashEmbedder`,
  :class:`OllamaEmbedder`, :func:`make_embedder` — embedding backends.
* :class:`ChromaIndex` — local persistent vector index.
* Three exception types: :class:`IndexingError`, :class:`EmbeddingError`,
  :class:`OfflineEmbeddingError`.

Importing this package is reasonably cheap: ``chromadb`` itself is
imported eagerly because the index is the focal class, but the
``DeterministicHashEmbedder`` path requires no network and no model
weights, so an offline test run pays only the chromadb import cost.
"""

from __future__ import annotations

from eurpe.retrieval.chunker import HierarchicalChunker, infer_section_type
from eurpe.retrieval.embeddings import (
    DeterministicHashEmbedder,
    Embedder,
    OllamaEmbedder,
    make_embedder,
)
from eurpe.retrieval.errors import (
    EmbeddingError,
    IndexingError,
    OfflineEmbeddingError,
)
from eurpe.retrieval.index import ChromaIndex
from eurpe.retrieval.models import Chunk
from eurpe.retrieval.retriever import (
    RetrievalPolicy,
    RetrievalResult,
    SourceStatusAwareRetriever,
)

__all__ = [
    "Chunk",
    "ChromaIndex",
    "DeterministicHashEmbedder",
    "Embedder",
    "EmbeddingError",
    "HierarchicalChunker",
    "IndexingError",
    "OfflineEmbeddingError",
    "OllamaEmbedder",
    "RetrievalPolicy",
    "RetrievalResult",
    "SourceStatusAwareRetriever",
    "infer_section_type",
    "make_embedder",
]
