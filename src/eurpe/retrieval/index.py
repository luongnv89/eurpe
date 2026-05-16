"""Local-only Chroma persistent vector index for proposal chunks.

Wraps :class:`chromadb.PersistentClient` with the EURPE conventions:

* **Telemetry off.** ``Settings(anonymized_telemetry=False)`` is passed
  at client construction so Chroma's startup probe does not phone
  home. The ``no_network`` test fixture would otherwise fail not on a
  query but at index *open*, which is confusing.
* **No default embedder.** ``embedding_function=None`` is passed when
  creating collections so Chroma does NOT instantiate its bundled
  ``DefaultEmbeddingFunction`` (which downloads ~80 MB of weights from
  Hugging Face on first use). EURPE supplies its own embeddings via
  the :class:`~eurpe.retrieval.embeddings.Embedder` Protocol.
* **Cosine space.** ``hnsw:space=cosine`` is set per collection.
  Combined with L2-normalised input vectors, similarity reduces to a
  dot product and ``similarity = 1 - distance`` (Chroma returns
  *distance*, not similarity).
* **Idempotent upsert.** Chunk identity is the
  :attr:`~eurpe.retrieval.models.Chunk.chunk_id` property — stable
  across runs — so re-ingesting a PDF replaces its previous chunks
  rather than duplicating them.
* **Flat metadata.** ``ChunkMetadata`` is nested but Chroma only
  accepts ``str | int | float | bool`` values in the metadata dict.
  ``_metadata_to_chroma`` flattens with dotted keys; the inverse,
  ``_chroma_to_metadata``, restores a fully-validated ``ChunkMetadata``
  on read. Both functions are tested for round-trip equality and for
  the ``None``-skip rule (Chroma rejects ``None``, so unset optional
  fields are simply omitted).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from eurpe.retrieval.embeddings import Embedder
from eurpe.retrieval.errors import IndexingError
from eurpe.retrieval.models import Chunk
from eurpe.schema import (
    ChunkMetadata,
    CitationAnchor,
    Programme,
    ProposalMetadata,
    SectionType,
    SourceStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata flattening helpers
# ---------------------------------------------------------------------------

#: Sentinel string written into Chroma when an optional field is unset.
#: Chroma rejects ``None`` values in the metadata dict, so we omit the key
#: entirely on write and detect "absent" on read by the missing key. The
#: constant exists so callers can compare against it instead of magic strings.
_NONE_MARKER = "__none__"


def _metadata_to_chroma(meta: ChunkMetadata) -> dict[str, str | int | float | bool]:
    """Flatten a :class:`ChunkMetadata` into a Chroma-compatible dict.

    The output uses dotted keys (``proposal.programme``, ``anchor.page``)
    so the structure round-trips losslessly through
    :func:`_chroma_to_metadata`. ``None`` values are *omitted* rather
    than written as the marker — Chroma's ``where`` filter cannot
    distinguish "field == marker" from "field absent" without extra
    work, and the inverse function treats a missing key as ``None``.
    Enum values are coerced to their underlying ``str`` so they survive
    Chroma's ``str | int | float | bool`` constraint.
    """

    proposal = meta.proposal
    anchor = meta.anchor

    flat: dict[str, str | int | float | bool] = {
        # ProposalMetadata fields
        "proposal.programme": str(proposal.programme.value),
        "proposal.call_id": proposal.call_id,
        "proposal.year": int(proposal.year),
        "proposal.outcome": str(proposal.outcome.value),
        "proposal.source_path": proposal.source_path,
        "proposal.language": proposal.language,
        # ChunkMetadata fields
        "section_type": str(meta.section_type.value),
        "chunk_index": int(meta.chunk_index),
        "source_status": str(meta.source_status.value),
        # CitationAnchor fields
        "anchor.document_id": anchor.document_id,
    }
    # Optional fields — only emit if set, so the round-trip preserves None.
    if proposal.topic_id is not None:
        flat["proposal.topic_id"] = proposal.topic_id
    if proposal.proposal_title is not None:
        flat["proposal.proposal_title"] = proposal.proposal_title
    if proposal.consortium_acronym is not None:
        flat["proposal.consortium_acronym"] = proposal.consortium_acronym
    if proposal.ingested_at is not None:
        # ISO 8601 string keeps the value sortable in Chroma's filter language.
        flat["proposal.ingested_at"] = proposal.ingested_at.isoformat()
    if proposal.content_hash is not None:
        # Optional sha256 hex; only written for chunks ingested after the
        # duplicate-detection feature landed. Older rows simply lack the
        # key, which is why ``find_by_content_hash`` returns ``[]`` for
        # them — see :meth:`ChromaIndex.find_by_content_hash`.
        flat["proposal.content_hash"] = proposal.content_hash
    if meta.parent_section_heading is not None:
        flat["parent_section_heading"] = meta.parent_section_heading
    if anchor.section_heading is not None:
        flat["anchor.section_heading"] = anchor.section_heading
    if anchor.page is not None:
        flat["anchor.page"] = int(anchor.page)
    if anchor.char_start is not None:
        flat["anchor.char_start"] = int(anchor.char_start)
    if anchor.char_end is not None:
        flat["anchor.char_end"] = int(anchor.char_end)
    return flat


def _chroma_to_metadata(d: dict[str, Any]) -> ChunkMetadata:
    """Reverse of :func:`_metadata_to_chroma`.

    Missing optional keys map back to ``None``. Enum-valued keys are
    coerced via the enum constructor so the final ``ChunkMetadata`` is
    validated end-to-end (drift validator included).
    """

    from datetime import datetime

    ingested_raw = d.get("proposal.ingested_at")
    ingested_at = datetime.fromisoformat(ingested_raw) if isinstance(ingested_raw, str) else None

    proposal = ProposalMetadata(
        programme=Programme(d["proposal.programme"]),
        call_id=str(d["proposal.call_id"]),
        topic_id=d.get("proposal.topic_id"),
        year=int(d["proposal.year"]),
        outcome=SourceStatus(d["proposal.outcome"]),
        proposal_title=d.get("proposal.proposal_title"),
        consortium_acronym=d.get("proposal.consortium_acronym"),
        source_path=str(d["proposal.source_path"]),
        language=str(d.get("proposal.language", "en")),
        ingested_at=ingested_at,
        content_hash=d.get("proposal.content_hash"),
    )
    anchor = CitationAnchor(
        document_id=str(d["anchor.document_id"]),
        section_heading=d.get("anchor.section_heading"),
        page=d.get("anchor.page"),
        char_start=d.get("anchor.char_start"),
        char_end=d.get("anchor.char_end"),
    )
    return ChunkMetadata(
        proposal=proposal,
        section_type=SectionType(d["section_type"]),
        parent_section_heading=d.get("parent_section_heading"),
        chunk_index=int(d["chunk_index"]),
        anchor=anchor,
        source_status=SourceStatus(d["source_status"]),
    )


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class ChromaIndex:
    """Persistent local Chroma index of :class:`Chunk` records.

    One :class:`ChromaIndex` instance == one collection. Multiple
    collections (e.g., per programme) live under the same on-disk
    directory; the client is shared across instances pointed at the
    same path.

    Cosine similarity is the project default (the chunker emits prose,
    L2-normalised vectors are well-suited). The collection's HNSW
    space is set at creation time via the ``metadata`` dict.

    Telemetry / network behaviour
    -----------------------------
    The constructor passes ``Settings(anonymized_telemetry=False)`` so
    Chroma does NOT phone home on startup, which is what makes the
    ``no_network`` test green. Collection creation passes
    ``embedding_function=None`` so Chroma does NOT instantiate its
    bundled MiniLM embedder (which would silently download weights).
    Both flags are non-negotiable for the offline-by-default contract.
    """

    def __init__(
        self,
        *,
        index_path: Path,
        embedder: Embedder,
        collection_name: str = "default",
    ) -> None:
        if not isinstance(index_path, Path):  # pragma: no cover - defensive
            raise TypeError(f"index_path must be a Path, got {type(index_path).__name__}")
        # Place all Chroma state under <index_path>/chroma so a future
        # second backend (FAISS, lance, ...) can coexist by using a
        # sibling subdirectory rather than fighting over the same dir.
        chroma_path = index_path / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        # ``allow_reset=True`` is intentionally NOT set: a stray
        # ``client.reset()`` would otherwise wipe a real index in dev.
        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedder = embedder
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # See module docstring: explicit None disables Chroma's
            # default HuggingFace-fetching embedder.
            embedding_function=None,
            # ``hnsw:space=cosine`` matches our L2-normalised vectors;
            # similarity = 1 - distance. The ``model`` key is recorded
            # for future migration scripts to detect embedder drift.
            metadata={
                "hnsw:space": "cosine",
                "embedder.model": embedder.model_name,
                "embedder.dimension": str(embedder.dimension),
            },
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def upsert(self, chunks: list[Chunk]) -> None:
        """Embed (where missing) and write chunks. Idempotent on chunk_id.

        Chroma's ``upsert`` semantically performs an INSERT-OR-REPLACE
        keyed on ``ids``, so calling this twice with the same chunks
        leaves the collection unchanged in count — which is the
        property the deterministic-rebuild tests rely on.

        Chunks with ``embedding is None`` are embedded in a single
        batch call; chunks that already carry a vector are passed
        through unchanged. Mixed batches are handled by re-using the
        cached vectors and only embedding the missing ones.
        """

        if not chunks:
            return

        # Validate dimension against the embedder before talking to
        # Chroma — a mismatched embedding would surface as an obscure
        # Chroma error otherwise.
        expected_dim = self._embedder.dimension
        # Embed only the chunks that need it (preserves caller-supplied
        # vectors if any are present, e.g., from a snapshot reload).
        missing_idx = [i for i, c in enumerate(chunks) if c.embedding is None]
        if missing_idx:
            try:
                fresh = self._embedder.embed([chunks[i].text for i in missing_idx])
            except Exception as exc:
                # Don't double-wrap if it's already an :class:`IndexingError`.
                if isinstance(exc, IndexingError):
                    raise
                raise IndexingError(f"Embedder failed during upsert: {exc}") from exc
            for slot, vec in zip(missing_idx, fresh, strict=True):
                if len(vec) != expected_dim:
                    raise IndexingError(
                        f"Embedder produced vector of width {len(vec)}; expected {expected_dim}."
                    )
                chunks[slot].embedding = vec

        # Build the four parallel arrays Chroma expects.
        ids = [c.chunk_id for c in chunks]
        embeddings = [c.embedding for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [_metadata_to_chroma(c.metadata) for c in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return the ``top_k`` chunks most similar to ``query_text``.

        Chroma returns *distance* (cosine in our case, range [0, 2]).
        We convert to *similarity* via ``1 - distance`` so callers see
        a familiar "higher is better" score in the documented range
        ``[-1, 1]`` (in practice, ``[0, 1]`` for L2-normalised
        non-negative vectors). The list is ordered by descending
        similarity (== ascending distance).

        ``where`` is forwarded to Chroma's filter language verbatim
        (e.g., ``{"source_status": "funded"}``). Use the same dotted
        keys :func:`_metadata_to_chroma` writes — see the integration
        tests for examples.
        """

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        try:
            (query_vec,) = self._embedder.embed([query_text])
        except Exception as exc:
            if isinstance(exc, IndexingError):
                raise
            raise IndexingError(f"Embedder failed during query: {exc}") from exc

        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
            include=["distances", "metadatas", "documents"],
        )
        # Chroma returns lists-of-lists keyed on the (single) query
        # vector. Unpack the first row.
        ids_row = result["ids"][0] if result.get("ids") else []
        distances_row = result["distances"][0] if result.get("distances") else []
        documents_row = result["documents"][0] if result.get("documents") else []
        metadatas_row = result["metadatas"][0] if result.get("metadatas") else []

        out: list[tuple[Chunk, float]] = []
        for _id, dist, doc, meta in zip(
            ids_row, distances_row, documents_row, metadatas_row, strict=True
        ):
            try:
                chunk_meta = _chroma_to_metadata(dict(meta))
            except Exception as exc:  # pragma: no cover - defensive
                # If a stored row's metadata fails the schema (e.g.,
                # written by an older version of EURPE), surface the
                # row id so the operator knows which to repair.
                raise IndexingError(
                    f"Stored metadata for chunk {_id!r} failed validation: {exc}"
                ) from exc
            chunk = Chunk(text=doc or "", metadata=chunk_meta)
            similarity = 1.0 - float(dist)
            out.append((chunk, similarity))
        return out

    def count(self) -> int:
        """Return the number of vectors stored in this collection."""

        return int(self._collection.count())

    # ------------------------------------------------------------------
    # duplicate-detection / incremental-indexing helpers
    # ------------------------------------------------------------------
    #
    # The methods below exist so the ingestion layer (HTTP route + CLI)
    # can answer three operator-facing questions before upserting:
    #
    # * "Has this PDF been indexed before?" — same bytes, same hash.
    # * "Have I seen this document under this filename before?" — by
    #   ``anchor.document_id`` (the PDF stem after archival).
    # * "Have I seen a different file with the same proposal_title and
    #   call_id?" — the soft duplicate that operators most often want a
    #   warning for.
    #
    # All three are read-only on the collection so they are safe to call
    # outside any transaction.

    def find_by_content_hash(self, content_hash: str) -> list[str]:
        """Return unique ``document_id``s of chunks whose proposal hash matches.

        Old chunks written before the content-hash field existed simply
        lack the ``proposal.content_hash`` key in their Chroma metadata
        row. Chroma's ``where`` filter matches only rows that *have* the
        key, so pre-feature data is invisible to this query — that is by
        design. Such rows become discoverable again the moment their
        owning proposal is re-ingested.
        """

        result = self._collection.get(
            where={"proposal.content_hash": content_hash},
            include=["metadatas"],
        )
        metadatas = result.get("metadatas") or []
        document_ids: list[str] = []
        seen: set[str] = set()
        for meta in metadatas:
            if not meta:
                continue
            doc_id = meta.get("anchor.document_id")
            if not isinstance(doc_id, str) or doc_id in seen:
                continue
            seen.add(doc_id)
            document_ids.append(doc_id)
        return document_ids

    def find_by_title_and_call(self, proposal_title: str | None, call_id: str) -> list[str]:
        """Return unique ``document_id``s matching ``(proposal_title, call_id)``.

        Returns ``[]`` early when ``proposal_title`` is falsy: the field is
        optional on :class:`ProposalMetadata`, and two title-less records
        in the same call would otherwise always collide and produce
        spurious soft-duplicate warnings.
        """

        if not proposal_title:
            return []
        result = self._collection.get(
            where={
                "$and": [
                    {"proposal.proposal_title": proposal_title},
                    {"proposal.call_id": call_id},
                ]
            },
            include=["metadatas"],
        )
        metadatas = result.get("metadatas") or []
        document_ids: list[str] = []
        seen: set[str] = set()
        for meta in metadatas:
            if not meta:
                continue
            doc_id = meta.get("anchor.document_id")
            if not isinstance(doc_id, str) or doc_id in seen:
                continue
            seen.add(doc_id)
            document_ids.append(doc_id)
        return document_ids

    def find_by_document_id(self, document_id: str) -> int:
        """Return the number of chunks stored under ``anchor.document_id``.

        Used by the dedup helper to detect the "corrected version" case
        (a new ingest whose archive name collides with an existing one)
        and by tests to assert delete-then-upsert left no orphans.
        """

        result = self._collection.get(
            where={"anchor.document_id": document_id},
            include=[],
        )
        ids = result.get("ids") or []
        return len(ids)

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete every chunk whose ``anchor.document_id`` matches and report the count.

        Returns the number of chunks removed (zero when nothing matched).
        The count is computed by diffing :meth:`find_by_document_id` before
        and after rather than relying on Chroma's delete response shape,
        which has shifted between versions.
        """

        before = self.find_by_document_id(document_id)
        if before == 0:
            return 0
        self._collection.delete(where={"anchor.document_id": document_id})
        return before

    def delete_collection(self) -> None:
        """Drop the collection. The on-disk database file is preserved.

        Useful in tests that want a clean slate without nuking the
        whole on-disk directory (which would also nuke any sibling
        collections under the same client).
        """

        self._client.delete_collection(self._collection_name)
