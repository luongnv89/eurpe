"""Shared pytest fixtures for EURPE.

Provides a ``no_network`` fixture that monkeypatches :func:`socket.socket.connect`
so that any code path attempting to reach the network during a test causes a
hard failure. Use it on tests that must prove offline-mode behaviour.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from eurpe.generation import DeterministicLLMClient, SectionGenerationWorkflow
from eurpe.retrieval import (
    ChromaIndex,
    Chunk,
    DeterministicHashEmbedder,
    RetrievalPolicy,
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


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test the moment any code tries to open a TCP connection."""

    def _blocked(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - intentional
        raise pytest.fail.Exception(
            "Network access attempted during a test marked offline-only."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)


@pytest.fixture
def deterministic_workflow(tmp_path: Path) -> SectionGenerationWorkflow:
    """Build a workflow with a real ChromaIndex + lenient policy + deterministic LLM.

    The lenient policy (``relevance_threshold=0.0``,
    ``max_rejected_fraction=1.0``) is critical: the
    :class:`DeterministicHashEmbedder` produces modest cosine scores
    that sit below the default 0.30 threshold, so without the override
    the workflow returns zero citations.
    """

    def _make_chunk(
        *,
        status: SourceStatus,
        programme: Programme = Programme.HORIZON_EUROPE,
        call_id: str = "HORIZON-CL5-2024-D3-02",
        section_type: SectionType = SectionType.METHODOLOGY,
        document_id: str = "doc",
        chunk_index: int = 0,
        text: str | None = None,
    ) -> Chunk:
        if text is None:
            text = (
                f"This is a sample {status.value} chunk discussing methodology "
                f"deep learning approach for the {programme.value} call."
            )
        proposal = ProposalMetadata(
            programme=programme,
            call_id=call_id,
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
            section_type=section_type,
            parent_section_heading="1.2 Methodology",
            chunk_index=chunk_index,
            anchor=anchor,
            source_status=status,
        )
        return Chunk(text=text, metadata=metadata)

    corpus = [
        _make_chunk(status=SourceStatus.FUNDED, document_id="f1", chunk_index=0),
        _make_chunk(status=SourceStatus.FUNDED, document_id="f2", chunk_index=1),
        _make_chunk(status=SourceStatus.FUNDED, document_id="f3", chunk_index=2),
        _make_chunk(
            status=SourceStatus.REJECTED,
            programme=Programme.HORIZON_2020,
            call_id="H2020-X-2018-1",
            document_id="r1",
            chunk_index=0,
        ),
        _make_chunk(
            status=SourceStatus.REJECTED,
            programme=Programme.HORIZON_2020,
            call_id="H2020-X-2018-1",
            document_id="r2",
            chunk_index=1,
        ),
    ]

    embedder = DeterministicHashEmbedder(dimension=128)
    index = ChromaIndex(
        index_path=tmp_path,
        embedder=embedder,
        collection_name="deterministic_workflow_fixture",
    )
    index.upsert(corpus)
    policy = RetrievalPolicy(
        relevance_threshold=0.0,
        max_rejected_fraction=1.0,
    )
    retriever = SourceStatusAwareRetriever(index, policy=policy)
    return SectionGenerationWorkflow(
        retriever=retriever,
        llm=DeterministicLLMClient(),
    )
