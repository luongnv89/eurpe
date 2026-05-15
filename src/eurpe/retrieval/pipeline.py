"""Reusable parse-to-index pipeline shared by the CLI and HTTP API.

Why this module exists
----------------------
The ``eurpe index build`` command and the upcoming ``POST /api/ingestion/confirm``
HTTP endpoint both need to perform the same chain of actions:

1. Take a :class:`~eurpe.ingestion.models.ParsedProposal` produced by Docling.
2. Join it with a :class:`~eurpe.schema.ProposalMetadata` record so the chunker
   can stamp programme / call / outcome onto every emitted chunk.
3. Hand the result to a :class:`~eurpe.retrieval.chunker.HierarchicalChunker`.
4. Upsert the resulting chunks into a :class:`~eurpe.retrieval.index.ChromaIndex`.

Before this helper existed the body of ``eurpe index build`` (see
``src/eurpe/retrieval/cli.py:148-222`` prior to issue #10) duplicated the
chain inline. The HTTP route would have either re-implemented it or imported
the Typer command, both of which are unhealthy. Lifting the chain into a
single function keeps the source-status drift validator
(:meth:`eurpe.schema.ChunkMetadata._status_matches_proposal`) on a single,
well-tested code path.

The helper deliberately stays silent — it returns the chunk count and lets
the caller (CLI / API) decide how to surface progress. Keeping it free of
``typer.echo`` calls is what lets the same function answer an HTTP request
and drive a Typer command without behavioural drift.
"""

from __future__ import annotations

from eurpe.ingestion.models import ParsedProposal
from eurpe.retrieval.chunker import HierarchicalChunker
from eurpe.retrieval.index import ChromaIndex
from eurpe.schema import ProposalMetadata


def index_proposal(
    parsed: ParsedProposal,
    proposal: ProposalMetadata,
    *,
    chunker: HierarchicalChunker,
    index: ChromaIndex,
) -> int:
    """Chunk a parsed proposal and upsert the chunks into the vector index.

    Parameters
    ----------
    parsed:
        Output of :meth:`eurpe.ingestion.docling_parser.DoclingProposalParser.parse`.
    proposal:
        The :class:`ProposalMetadata` that supplies programme / call / outcome
        for every emitted chunk. The chunker enforces
        ``chunk.source_status == proposal.outcome`` via the drift validator
        on :class:`~eurpe.schema.ChunkMetadata` — this helper does not need
        to re-check it, but it does centralise the *one* code path where the
        invariant matters.
    chunker:
        Pre-configured :class:`HierarchicalChunker`. Sharing one instance
        across many proposals is cheap because the chunker itself is
        stateless aside from its tuning knobs.
    index:
        Open :class:`ChromaIndex` to upsert into. Idempotent on
        :attr:`Chunk.chunk_id`, so re-calling this function with the same
        ``(parsed, proposal)`` leaves the collection in the same state.

    Returns
    -------
    int
        Number of chunks emitted by the chunker for ``parsed``. Equal to the
        size of the upsert batch, useful for status logging at the call site.

    Notes
    -----
    No I/O happens before the upsert: if the chunker returns an empty list
    the helper returns 0 without touching the index, matching the
    short-circuit at :meth:`ChromaIndex.upsert`.
    """

    chunks = chunker.chunk(parsed, proposal)
    index.upsert(chunks)
    return len(chunks)
