"""Retrieval-layer models.

Hosts the :class:`Chunk` record that the chunker produces and the index
consumes. ``Chunk`` is intentionally minimal: text + the existing
:class:`~eurpe.schema.ChunkMetadata` + an optional embedding vector. All
provenance and source-status information already lives inside
``ChunkMetadata``; we do NOT re-thread programme/call/source_status
fields through ``Chunk`` because that duplication is the exact failure
mode the schema layer's drift validator was built to prevent.

The :attr:`Chunk.chunk_id` property is what makes upserts idempotent
across runs. It composes the document id with the parent section
heading hash and the in-document chunk index, so re-ingesting the same
PDF replaces previous chunks rather than duplicating them.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from eurpe.schema import ChunkMetadata


class Chunk(BaseModel):
    """A retrieval-ready piece of a parsed proposal.

    The text is what gets embedded and indexed; ``metadata`` carries the
    full provenance (programme, call, outcome, source_status, page,
    char offsets) that downstream filtering and citation rendering rely
    on. ``embedding`` is left optional so the chunker can build chunks
    without holding the embedder model in scope — the index call site
    embeds at upsert time.

    ``extra="forbid"`` keeps typos in field names loud rather than
    silently dropping unrecognised keys; matches the convention used
    across :mod:`eurpe.schema` and :mod:`eurpe.ingestion`.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        description="Plain text of the chunk; what the embedder consumes.",
    )
    metadata: ChunkMetadata = Field(
        description="Per-chunk provenance and retrieval-facing labels.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description=(
            "Optional dense vector. Set by the index at upsert time if "
            "left None at construction; preserved across model dumps so "
            "an index can be rebuilt from a JSON snapshot."
        ),
    )

    @property
    def chunk_id(self) -> str:
        """Stable, deterministic identifier used for upsert/dedup.

        Format::

            {document_id}::{section_hash}::{chunk_index}

        ``document_id`` is taken verbatim from
        ``metadata.anchor.document_id``. ``section_hash`` is the first
        12 hex chars of a SHA-256 digest of
        ``metadata.parent_section_heading`` (or the literal ``"__no_section__"``
        when no heading is set). ``chunk_index`` is
        ``metadata.chunk_index`` formatted as zero-padded base-10.

        The hash collapses long heading strings into a fixed-length
        token while staying deterministic, so re-ingesting the same
        section produces the same id and the index upsert acts as
        replace-in-place. Twelve hex chars = 48 bits ≈ 281 trillion
        slots; collisions are not a concern at proposal scale.
        """

        anchor_doc = self.metadata.anchor.document_id
        heading = self.metadata.parent_section_heading or "__no_section__"
        section_hash = hashlib.sha256(heading.encode("utf-8")).hexdigest()[:12]
        return f"{anchor_doc}::{section_hash}::{self.metadata.chunk_index:06d}"
