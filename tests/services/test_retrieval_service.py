"""Tests for :class:`eurpe.retrieval.RetrievalService`.

Builds a small in-memory Chroma index, wires a retriever through the
service, and verifies request/response shapes + error propagation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eurpe.retrieval import (
    ChromaIndex,
    Chunk,
    DeterministicHashEmbedder,
    RetrievalPolicy,
    RetrievalQuery,
    RetrievalService,
    SourceStatusAwareRetriever,
)
from eurpe.schema import (
    ChunkMetadata,
    CitationAnchor,
    Programme,
    ProposalMetadata,
    SectionType,
    SourceStatus,
)


def _build_chunk(
    *,
    document_id: str,
    chunk_index: int,
    status: SourceStatus = SourceStatus.FUNDED,
) -> Chunk:
    proposal = ProposalMetadata(
        programme=Programme.HORIZON_EUROPE,
        call_id="HORIZON-CL5-2024-D3-02",
        year=2024,
        outcome=status,
        proposal_title=f"Proposal {document_id}",
        source_path=f"data/{document_id}.pdf",
    )
    anchor = CitationAnchor(
        document_id=document_id,
        section_heading="1.2 Methodology",
        page=8,
    )
    metadata = ChunkMetadata(
        proposal=proposal,
        section_type=SectionType.METHODOLOGY,
        parent_section_heading="1.2 Methodology",
        chunk_index=chunk_index,
        anchor=anchor,
        source_status=status,
    )
    return Chunk(
        text=f"Methodology content for {document_id} index {chunk_index}.",
        metadata=metadata,
    )


@pytest.fixture
def retrieval_service(tmp_path) -> RetrievalService:
    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="retrieval_service_tests",
    )
    index.upsert(
        [
            _build_chunk(document_id="f1", chunk_index=0),
            _build_chunk(document_id="f2", chunk_index=0),
        ]
    )
    retriever = SourceStatusAwareRetriever(
        index, policy=RetrievalPolicy(relevance_threshold=0.0)
    )
    return RetrievalService(retriever)


def test_retrieval_service_returns_response_with_results(retrieval_service) -> None:
    """Happy path: query returns results and ``result_count`` matches."""

    response = retrieval_service.query(
        RetrievalQuery(query="methodology content", top_k=5)
    )
    assert response.result_count == len(response.results)
    assert response.result_count >= 1
    # AC #2: each result carries its source-status label.
    for result in response.results:
        assert result.source_status is SourceStatus.FUNDED


def test_retrieval_service_query_rejects_empty_string() -> None:
    """Pydantic validates the request — empty query strings fail at the boundary."""

    with pytest.raises(ValidationError):
        RetrievalQuery(query="", top_k=5)


def test_retrieval_service_query_rejects_zero_top_k() -> None:
    """``top_k`` has a positive lower bound; bad inputs surface as ValidationError."""

    with pytest.raises(ValidationError):
        RetrievalQuery(query="x", top_k=0)


def test_retrieval_service_empty_index_returns_no_results(tmp_path) -> None:
    """An empty index produces an empty response without raising.

    The error path the AC wants here is "no matches" rather than a
    crash. The service exposes that as ``result_count == 0``.
    """

    embedder = DeterministicHashEmbedder(dimension=64)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="retrieval_service_empty_index",
    )
    retriever = SourceStatusAwareRetriever(
        index, policy=RetrievalPolicy(relevance_threshold=0.0)
    )
    service = RetrievalService(retriever)
    response = service.query(RetrievalQuery(query="anything", top_k=5))
    assert response.result_count == 0
    assert response.results == []
