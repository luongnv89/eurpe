"""Direct unit tests for :func:`eurpe.retrieval.pipeline.index_proposal`.

The CLI tests exercise the helper indirectly through ``eurpe index build``;
these tests pin the helper's contract directly so a future refactor of the
CLI cannot mask a regression in the helper itself. Three things are
load-bearing:

1. The returned chunk count matches the actual number of chunks the
   chunker emitted for the input. The CLI uses this number to print
   progress; a wrong return value would silently break the operator's
   feedback loop.
2. Every upserted chunk's ``source_status`` equals the proposal's
   ``outcome``. The drift validator on
   :class:`eurpe.schema.ChunkMetadata` catches mismatches by raising,
   so the helper does not need to re-check, but we assert the
   downstream-visible value is what the operator confirmed.
3. The upsert is idempotent on chunk_id. Calling :func:`index_proposal`
   twice on the same input leaves the collection at the same count;
   this is what powers re-indexing without duplicates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eurpe.ingestion.models import ParsedProposal, ParsedSection
from eurpe.retrieval import (
    ChromaIndex,
    DeterministicHashEmbedder,
    HierarchicalChunker,
    index_proposal,
)
from eurpe.schema import Programme, ProposalMetadata, SourceStatus


def _build_parsed(source_path: str = "/tmp/example.pdf") -> ParsedProposal:
    """Return a small parsed proposal that produces > 1 chunk."""

    return ParsedProposal(
        source_path=source_path,
        title="Example",
        sections=[
            ParsedSection(
                heading="1. Excellence",
                level=1,
                text=(
                    "The proposal pioneers a novel deep-learning method for "
                    "anomaly detection in network telemetry. Methodology "
                    "emphasises reproducibility, open data, and federated "
                    "training across consortium partners."
                ),
                page_start=1,
                page_end=1,
            ),
            ParsedSection(
                heading="2. Impact",
                level=1,
                text=(
                    "Expected impact spans operational uplift for SMEs and "
                    "policy contributions to NIS2 compliance. Dissemination "
                    "plans target three industry conferences and one open "
                    "source release per year."
                ),
                page_start=2,
                page_end=2,
            ),
        ],
        page_count=2,
        parser="stub",
    )


def _build_proposal(outcome: SourceStatus = SourceStatus.FUNDED) -> ProposalMetadata:
    return ProposalMetadata(
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL3-2024-CS-01",
        topic_id="883588",
        year=2024,
        outcome=outcome,
        proposal_title="Example",
        consortium_acronym="EXP",
        source_path="/tmp/example.pdf",
    )


@pytest.fixture
def in_memory_index(tmp_path: Path) -> ChromaIndex:
    embedder = DeterministicHashEmbedder(dimension=128)
    return ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="pipeline_test",
    )


def test_index_proposal_returns_chunk_count(in_memory_index: ChromaIndex) -> None:
    """Return value must equal the size of the upsert batch."""

    chunker = HierarchicalChunker()
    parsed = _build_parsed()
    proposal = _build_proposal()
    count = index_proposal(parsed, proposal, chunker=chunker, index=in_memory_index)
    assert count >= 2  # one chunk per section at minimum
    assert count == in_memory_index.count()


@pytest.mark.parametrize(
    "outcome",
    [SourceStatus.FUNDED, SourceStatus.REJECTED, SourceStatus.ESR_NOTE, SourceStatus.UNKNOWN],
)
def test_index_proposal_stamps_proposal_outcome_on_every_chunk(
    in_memory_index: ChromaIndex, outcome: SourceStatus
) -> None:
    """Every upserted chunk inherits ``proposal.outcome`` as its source_status."""

    chunker = HierarchicalChunker()
    parsed = _build_parsed()
    proposal = _build_proposal(outcome=outcome)
    index_proposal(parsed, proposal, chunker=chunker, index=in_memory_index)

    # Query without filters; the deterministic embedder makes ranking
    # uninteresting but every chunk MUST carry the same source_status.
    results = in_memory_index.query("methodology", top_k=20)
    assert results, "query returned no chunks"
    for chunk, _score in results:
        assert chunk.metadata.source_status is outcome
        assert chunk.metadata.proposal.outcome is outcome


def test_index_proposal_is_idempotent_on_repeated_calls(
    in_memory_index: ChromaIndex,
) -> None:
    """Calling the helper twice on the same input does not double-count chunks.

    Idempotency is a property of the underlying ``ChromaIndex.upsert`` keyed
    on :attr:`Chunk.chunk_id`, but the helper must not transform inputs in
    ways that break id stability. Pinning it here makes a future regression
    visible at the helper level, not 30 layers down.
    """

    chunker = HierarchicalChunker()
    parsed = _build_parsed()
    proposal = _build_proposal()

    first = index_proposal(parsed, proposal, chunker=chunker, index=in_memory_index)
    after_first = in_memory_index.count()
    second = index_proposal(parsed, proposal, chunker=chunker, index=in_memory_index)
    after_second = in_memory_index.count()

    assert first == second
    assert after_first == after_second
